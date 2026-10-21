from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class FieldCandidate:
    field: str
    value: str | None
    confidence: float | None
    evidence: str
    page: int


class ExtractionResult(BaseModel):
    field: str
    value: str | None
    confidence: float | None = Field(ge=0, le=1)
    evidence: str
    page: int | None


class PipelineMetrics(BaseModel):
    total_seconds: float
    text_extraction_seconds: float
    model_load_seconds: float
    inference_seconds: float
    fields_requested: int
    fields_extracted: int


class ExtractionResponse(BaseModel):
    model: str
    pages_processed: int
    metrics: PipelineMetrics
    results: list[ExtractionResult]


def grounded_candidate(
    *, field: str, value: object, confidence: object, evidence: object, page: int, page_text: str
) -> FieldCandidate | None:
    """Accept only model answers that quote text actually present on the page."""
    if not isinstance(value, str) or not value.strip() or not isinstance(evidence, str) or not evidence.strip():
        return None
    if evidence not in page_text:
        return None
    normalized_value = value.strip().casefold()
    normalized_evidence = evidence.casefold()
    if normalized_value not in normalized_evidence:
        return None
    if confidence is None:
        score = None
    else:
        try:
            score = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            return None
    return FieldCandidate(
        field=field,
        value=value.strip(),
        confidence=score,
        evidence=evidence.strip(),
        page=page,
    )


def select_best(candidates: list[FieldCandidate]) -> FieldCandidate | None:
    return max(candidates, key=lambda item: item.confidence if item.confidence is not None else 0, default=None)
