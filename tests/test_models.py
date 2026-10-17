from app.models import grounded_candidate, select_best
from app.main import parse_fields


def test_grounded_candidate_requires_verbatim_evidence_and_value():
    text = "Invoice number: INV-4421\nTotal: $99.00"
    candidate = grounded_candidate(
        field="Invoice number", value="INV-4421", confidence=0.92,
        evidence="Invoice number: INV-4421", page=1, page_text=text,
    )
    assert candidate is not None
    assert candidate.confidence == 0.92
    assert grounded_candidate(
        field="Invoice number", value="INV-9999", confidence=1,
        evidence="Invoice number: INV-9999", page=1, page_text=text,
    ) is None


def test_select_best_and_field_parsing():
    low = grounded_candidate(field="date", value="2024-01-01", confidence=.4, evidence="2024-01-01", page=1, page_text="2024-01-01")
    high = grounded_candidate(field="date", value="2024-01-02", confidence=.8, evidence="2024-01-02", page=2, page_text="2024-01-02")
    assert select_best([low, high]) == high
    assert parse_fields("Date\n\nTotal\nDate\n") == ["Date", "Total"]
