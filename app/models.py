from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class FieldCandidate:
    field: str
    value: str | None
    confidence: float
    evidence: str
    page: int


class ExtractionResult(BaseModel):
    field: str
    value: str | None
    confidence: float = Field(ge=0, le=1)
    evidence: str
    page: int | None


class ExtractionResponse(BaseModel):
    model: str
    pages_processed: int
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
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return None
    return FieldCandidate(
        field=field,
        value=value.strip(),
        confidence=max(0.0, min(1.0, score)),
        evidence=evidence.strip(),
        page=page,
    )


def select_best(candidates: list[FieldCandidate]) -> FieldCandidate | None:
    return max(candidates, key=lambda item: item.confidence, default=None)
