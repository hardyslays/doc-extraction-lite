from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.extractors import ExtractedSpan
from app.main import app, get_extractor


class FakeExtractor:
    name = "fake-extractor"

    def extract(self, text, fields):
        if "INV-4421" not in text:
            return []
        start = text.index("INV-4421")
        return [
            ExtractedSpan(
                field="Invoice number",
                value=text[start:start + len("INV-4421")],
                confidence=0.93,
                start=start,
                end=start + len("INV-4421"),
            )
        ]


def text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extract_endpoint_returns_grounded_results_and_missing_fields():
    app.dependency_overrides[get_extractor] = lambda: FakeExtractor()
    try:
        response = TestClient(app).post(
            "/api/extract",
            files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
            data={"fields": "Invoice number\nDue date"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "model": "fake-extractor",
        "pages_processed": 1,
        "results": [
            {
                "field": "Invoice number",
                "value": "INV-4421",
                "confidence": 0.93,
                "evidence": "INV-4421",
                "page": 1,
            },
            {
                "field": "Due date",
                "value": None,
                "confidence": 0.0,
                "evidence": "",
                "page": None,
            },
        ],
    }


def test_extract_endpoint_publishes_response_schema():
    response_schema = app.openapi()["paths"]["/api/extract"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert response_schema == {"$ref": "#/components/schemas/ExtractionResponse"}
