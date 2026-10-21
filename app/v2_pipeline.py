"""The v2 extraction transaction executed in the isolated worker."""

import resource
from time import perf_counter
from typing import Any

from app.nuextract import load_extractor
from app.v2_document import parse_pdf
from app.v2_extract import extract_structured
from app.v2_schema import blank_data


def _missing_count(template: Any, data: Any) -> int:
    if isinstance(template, str):
        return int(data is None)
    if isinstance(template, list):
        return int(data is None)
    return sum(_missing_count(child, (data or {}).get(key)) for key, child in template.items())


def run_v2_pipeline(pdf_bytes: bytes, template: dict, model: str, force_ocr: bool = False,
                    parser: str = "docling") -> dict:
    started = perf_counter()
    document = parse_pdf(pdf_bytes, force_ocr=force_ocr, parser=parser)
    model_started = perf_counter()
    if any(page.blocks for page in document.pages):
        load_extractor(model)
    model_load_seconds = perf_counter() - model_started
    inference_started = perf_counter()
    if any(page.blocks for page in document.pages):
        data, evidence, calls, warnings = extract_structured(document, template, model)
    else:
        data, evidence, calls, warnings = blank_data(template), {}, 0, []
    inference_seconds = perf_counter() - inference_started
    missing = _missing_count(template, data)
    failed_pages = [page.number for page in document.pages if page.route in {"ocr_failed", "mixed_partial"}]
    if failed_pages:
        warnings.append("Structured OCR returned no text for page(s): " + ", ".join(map(str, failed_pages)))
    return {
        "model": model,
        "parser": parser,
        "pages_processed": len(document.pages),
        "complete": not failed_pages and not any(warning.startswith("Only ") for warning in warnings),
        "data": data,
        "evidence": evidence,
        "warnings": warnings,
        "metrics": {
            "total_seconds": perf_counter() - started,
            "native_parse_seconds": document.native_seconds,
            "ocr_seconds": document.ocr_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "model_calls": calls,
            "abstentions": missing,
            "page_routes": {str(page.number): page.route for page in document.pages},
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
