"""SQLite-backed async job queue with idempotency keys, retries, and extraction worker."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, col, select

from docextract.config import Settings, get_settings
from docextract.extract.llm import ExtractionLLM
from docextract.extract.runner import extract_structured_document
from docextract.scoring.confidence import score_document_confidence
from docextract.store.models import DocumentRecord, DocumentStatus
from docextract.store.repository import DocumentRepository
from docextract.validate.rules import RuleResult, Severity, ValidationReport, validate_document
from docextract.vision.ocr import OcrEngine, extract_document_layout


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:14]}"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExtractionJob(SQLModel, table=True):
    """Persisted background extraction job row with idempotency key and retry tracking."""

    __tablename__ = "extraction_job"

    id: str = Field(default_factory=_new_job_id, primary_key=True)
    document_id: str = Field(index=True)
    idempotency_key: str | None = Field(default=None, index=True, unique=True)
    status: str = Field(default=JobStatus.PENDING.value, index=True)
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)
    last_error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class JobQueue:
    """Persistent job queue coordinating document extraction workers."""

    def __init__(
        self,
        repo: DocumentRepository,
        *,
        llm: ExtractionLLM | None = None,
        ocr_engine: OcrEngine | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.repo = repo
        self.llm = llm
        self.ocr_engine = ocr_engine
        self.settings = settings or get_settings()
        SQLModel.metadata.create_all(self.repo.engine)

    def enqueue(
        self,
        document_id: str,
        *,
        idempotency_key: str | None = None,
        max_attempts: int = 3,
    ) -> ExtractionJob:
        """Enqueue an extraction job or return the existing job if `idempotency_key` matches."""
        with Session(self.repo.engine) as session:
            if idempotency_key:
                stmt = select(ExtractionJob).where(
                    ExtractionJob.idempotency_key == idempotency_key
                )
                existing = session.exec(stmt).first()
                if existing is not None:
                    return existing

            job = ExtractionJob(
                document_id=document_id,
                idempotency_key=idempotency_key,
                status=JobStatus.PENDING.value,
                attempts=0,
                max_attempts=max_attempts,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def get_job(self, job_id: str) -> ExtractionJob | None:
        with Session(self.repo.engine) as session:
            return session.get(ExtractionJob, job_id)

    def get_job_for_document(self, document_id: str) -> ExtractionJob | None:
        with Session(self.repo.engine) as session:
            stmt = (
                select(ExtractionJob)
                .where(ExtractionJob.document_id == document_id)
                .order_by(col(ExtractionJob.created_at).desc())
            )
            return session.exec(stmt).first()

    def _load_raw_bytes(self, rec: DocumentRecord) -> bytes:
        doc_dir = self.settings.storage_dir / rec.doc_sha256[:2] / rec.doc_sha256
        candidates = list(doc_dir.glob("source.*"))
        if not candidates:
            raise FileNotFoundError(
                f"Raw source file missing for document '{rec.id}' in {doc_dir.as_posix()}"
            )
        return Path(candidates[0]).read_bytes()

    async def process_job(self, job_id: str) -> ExtractionJob:
        """Execute a single queued extraction job with automatic retry eligibility on failure."""
        with Session(self.repo.engine) as session:
            job = session.get(ExtractionJob, job_id)
            if job is None:
                raise KeyError(f"Job '{job_id}' not found")
            if job.status == JobStatus.SUCCEEDED.value:
                return job

            job.status = JobStatus.RUNNING.value
            job.attempts += 1
            job.updated_at = _utc_now()
            session.add(job)
            session.commit()
            session.refresh(job)
            doc_id = job.document_id
            attempt_num = job.attempts
            max_attempts = job.max_attempts

        try:
            rec = self.repo.get(doc_id)
            raw_bytes = self._load_raw_bytes(rec)
            layouts = extract_document_layout(
                raw_bytes,
                ocr_engine=self.ocr_engine,
                target_dpi=self.settings.target_dpi,
            )
            ext_out = await extract_structured_document(
                layouts,
                llm=self.llm,
                settings=self.settings,
            )
            val_report = validate_document(ext_out.document, settings=self.settings)

            # Cross-document duplicate invoice check
            if ext_out.document.kind.value == "invoice":
                from docextract.schemas.document import Invoice

                assert isinstance(ext_out.document, Invoice)
                dup = self.repo.find_duplicate_invoice(
                    ext_out.document.vendor.name,
                    ext_out.document.invoice_number,
                    exclude_doc_id=doc_id,
                )
                if dup is not None:
                    dup_rule = RuleResult(
                        rule_id="duplicate_vendor_invoice",
                        passed=False,
                        severity=Severity.ERROR,
                        field_path="invoice_number",
                        expected="unique per vendor",
                        actual=ext_out.document.invoice_number,
                        message=(
                            f"Duplicate invoice '{ext_out.document.invoice_number}' "
                            f"for vendor '{ext_out.document.vendor.name}' (matches {dup.id})"
                        ),
                    )
                    new_results = [*val_report.results, dup_rule]
                    val_report = ValidationReport(
                        passed=False,
                        error_count=val_report.error_count + 1,
                        warning_count=val_report.warning_count,
                        results=new_results,
                    )

            conf_report = score_document_confidence(
                ext_out.document,
                layouts=layouts,
                validation=val_report,
                settings=self.settings,
            )
            self.repo.save_extraction(doc_id, ext_out.document, val_report, conf_report)

            with Session(self.repo.engine) as session:
                job_done = session.get(ExtractionJob, job_id)
                assert job_done is not None
                job_done.status = JobStatus.SUCCEEDED.value
                job_done.last_error = None
                job_done.updated_at = _utc_now()
                session.add(job_done)
                session.commit()
                session.refresh(job_done)
                return job_done

        except Exception as exc:
            err_str = str(exc)
            terminal = attempt_num >= max_attempts
            with Session(self.repo.engine) as session:
                job_err = session.get(ExtractionJob, job_id)
                assert job_err is not None
                job_err.status = JobStatus.FAILED.value if terminal else JobStatus.PENDING.value
                job_err.last_error = err_str
                job_err.updated_at = _utc_now()
                session.add(job_err)
                session.commit()
                session.refresh(job_err)

            if terminal:
                rec_now = self.repo.get(doc_id)
                if rec_now.status == DocumentStatus.RECEIVED.value:
                    self.repo.mark_failed(doc_id, err_str)
            return job_err

    async def process_next_pending(self) -> ExtractionJob | None:
        """Pick the oldest `pending` job and process it."""
        with Session(self.repo.engine) as session:
            stmt = (
                select(ExtractionJob)
                .where(ExtractionJob.status == JobStatus.PENDING.value)
                .order_by(col(ExtractionJob.created_at).asc())
            )
            next_job = session.exec(stmt).first()
            if next_job is None:
                return None
            job_id = next_job.id
        return await self.process_job(job_id)

    async def drain_pending(self, max_jobs: int = 50) -> list[ExtractionJob]:
        """Process pending jobs until the queue is empty or `max_jobs` iterations complete."""
        processed: list[ExtractionJob] = []
        for _ in range(max_jobs):
            res = await self.process_next_pending()
            if res is None:
                break
            processed.append(res)
            await asyncio.sleep(0)
        return processed
