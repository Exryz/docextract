"""Public exports for `datagen`: synthetic invoice/receipt generator and fault injector."""

from datagen.generator import (
    FaultType,
    GeneratedSample,
    LayoutStyle,
    generate_dataset,
    generate_sample,
    inject_fault,
    render_invoice_pdf,
    render_receipt_pdf,
)

__all__ = [
    "FaultType",
    "GeneratedSample",
    "LayoutStyle",
    "generate_dataset",
    "generate_sample",
    "inject_fault",
    "render_invoice_pdf",
    "render_receipt_pdf",
]
