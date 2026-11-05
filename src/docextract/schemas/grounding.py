"""Bounding box, word token, and field grounding schemas."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class BoundingBox(BaseModel):
    """Normalized page bounding box in `[0.0, 1.0]` (`x0, y0` top-left, `x1, y1` bottom-right)."""

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1, description="1-based page number")
    x0: float = Field(ge=0.0, le=1.0, description="Left normalized coordinate")
    y0: float = Field(ge=0.0, le=1.0, description="Top normalized coordinate")
    x1: float = Field(ge=0.0, le=1.0, description="Right normalized coordinate")
    y1: float = Field(ge=0.0, le=1.0, description="Bottom normalized coordinate")

    @model_validator(mode="after")
    def _validate_bounds(self) -> BoundingBox:
        if self.x1 < self.x0:
            raise ValueError(f"x1 ({self.x1}) must be >= x0 ({self.x0})")
        if self.y1 < self.y0:
            raise ValueError(f"y1 ({self.y1}) must be >= y0 ({self.y0})")
        return self

    @property
    def width(self) -> float:
        return round(self.x1 - self.x0, 6)

    @property
    def height(self) -> float:
        return round(self.y1 - self.y0, 6)

    @property
    def area(self) -> float:
        return round((self.x1 - self.x0) * (self.y1 - self.y0), 6)

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection-over-Union with another BoundingBox on the same page."""
        if self.page != other.page:
            return 0.0
        inter_x0 = max(self.x0, other.x0)
        inter_y0 = max(self.y0, other.y0)
        inter_x1 = min(self.x1, other.x1)
        inter_y1 = min(self.y1, other.y1)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return round(inter_area / union_area, 6)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def to_pixel_rect(self, page_width: int, page_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to `(left, top, right, bottom)` pixel integers."""
        return (
            int(round(self.x0 * page_width)),
            int(round(self.y0 * page_height)),
            int(round(self.x1 * page_width)),
            int(round(self.y1 * page_height)),
        )

    @classmethod
    def from_pixel_rect(
        cls,
        page: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        page_width: float,
        page_height: float,
    ) -> BoundingBox:
        if page_width <= 0 or page_height <= 0:
            raise ValueError("page_width and page_height must be positive")
        nx0 = max(0.0, min(1.0, x0 / page_width))
        ny0 = max(0.0, min(1.0, y0 / page_height))
        nx1 = max(nx0, min(1.0, x1 / page_width))
        ny1 = max(ny0, min(1.0, y1 / page_height))
        return cls(
            page=page,
            x0=round(nx0, 6),
            y0=round(ny0, 6),
            x1=round(nx1, 6),
            y1=round(ny1, 6),
        )

    @classmethod
    def merge(cls, boxes: list[BoundingBox]) -> BoundingBox:
        """Return the minimal bounding box enclosing all boxes on the same page."""
        if not boxes:
            raise ValueError("Cannot merge an empty list of BoundingBoxes")
        first_page = boxes[0].page
        if any(b.page != first_page for b in boxes):
            raise ValueError("All BoundingBoxes in merge() must belong to the same page")
        return cls(
            page=first_page,
            x0=min(b.x0 for b in boxes),
            y0=min(b.y0 for b in boxes),
            x1=max(b.x1 for b in boxes),
            y1=max(b.y1 for b in boxes),
        )


class WordToken(BaseModel):
    """A single word extracted from a PDF text layer or OCR engine with its bounding box."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BoundingBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GroundedValue(BaseModel, Generic[T]):
    """Wraps an extracted value with its page bounding box, raw source text, and confidence."""

    model_config = ConfigDict(frozen=True)

    value: T
    source_bbox: BoundingBox | None = None
    raw_text: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
