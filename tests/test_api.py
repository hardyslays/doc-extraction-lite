from io import BytesIO
from threading import Event
from time import monotonic, sleep

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.extractors import ExtractedSpan
from app.main import app, jobs
from app.nuextract import NUEXTRACT3_MODEL, TINY_MODEL, LocalModelError


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


def test_extract_endpoint_returns_grounded_results_and_missing_fields(monkeypatch):
    monkeypatch.setattr("app.main.get_extractor", lambda model: FakeExtractor())
    response = TestClient(app).post(
        "/api/extract",
        files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
        data={"fields": "Invoice number\nDue date"},
    )

    assert response.status_code == 200
    body = response.json()
    metrics = body.pop("metrics")
    assert body == {
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
                "confidence": None,
                "evidence": "",
                "page": None,
            },
        ],
    }
    assert metrics["fields_requested"] == 2
    assert metrics["fields_extracted"] == 1
    assert metrics["total_seconds"] >= metrics["inference_seconds"] >= 0
    assert metrics["total_seconds"] >= metrics["text_extraction_seconds"] >= 0


def test_extract_endpoint_publishes_response_schema():
    response_schema = app.openapi()["paths"]["/api/extract"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert response_schema == {"$ref": "#/components/schemas/ExtractionResponse"}


def test_extract_endpoint_accepts_force_ocr(monkeypatch):
    captured = []

    def fake_extract_pages(_, force_ocr=False):
        captured.append(force_ocr)
        return ["Invoice number: INV-4421"]

    monkeypatch.setattr("app.main.extract_pages", fake_extract_pages)
    monkeypatch.setattr("app.main.get_extractor", lambda model: FakeExtractor())
    response = TestClient(app).post(
        "/api/extract",
        files={"pdf": ("invoice.pdf", text_pdf("ignored"), "application/pdf")},
        data={"fields": "Invoice number", "force_ocr": "true"},
    )

    assert response.status_code == 200
    assert captured == [True]


def test_extract_endpoint_selects_each_model_and_rejects_unknown(monkeypatch):
    selected = []

    def fake_get_extractor(model):
        selected.append(model)
        extractor = FakeExtractor()
        extractor.name = model
        return extractor

    monkeypatch.setattr("app.main.get_extractor", fake_get_extractor)
    client = TestClient(app)
    for model in (TINY_MODEL, NUEXTRACT3_MODEL):
        response = client.post(
            "/api/extract",
            files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
            data={"fields": "Invoice number", "model": model},
        )
        assert response.status_code == 200
        assert response.json()["model"] == model
    assert selected == [TINY_MODEL, NUEXTRACT3_MODEL]

    invalid = client.post(
        "/api/extract",
        files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
        data={"fields": "Invoice number", "model": "not-a-model"},
    )
    assert invalid.status_code == 422
    assert selected == [TINY_MODEL, NUEXTRACT3_MODEL]


def test_extract_endpoint_maps_value_to_second_page(monkeypatch):
    monkeypatch.setattr("app.main.extract_pages", lambda content, force_ocr=False: ["Cover page", "Invoice number: INV-4421"])
    monkeypatch.setattr("app.main.get_extractor", lambda model: FakeExtractor())
    response = TestClient(app).post(
        "/api/extract",
        files={"pdf": ("invoice.pdf", text_pdf("ignored"), "application/pdf")},
        data={"fields": "Invoice number"},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["page"] == 2


def test_index_lists_both_models():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert TINY_MODEL in response.text
    assert NUEXTRACT3_MODEL in response.text


def test_inference_failure_returns_json_error(monkeypatch):
    class FailingExtractor:
        name = NUEXTRACT3_MODEL

        def extract(self, text, fields):
            raise LocalModelError("NuExtract3 inference failed: out of memory")

    monkeypatch.setattr("app.main.get_extractor", lambda model: FailingExtractor())
    response = TestClient(app).post(
        "/api/extract",
        files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
        data={"fields": "Invoice number", "model": NUEXTRACT3_MODEL},
    )

    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    assert "out of memory" in response.json()["detail"]


def test_job_returns_quickly_then_completes_without_a_long_http_request(monkeypatch):
    release = Event()

    class SlowExtractor(FakeExtractor):
        def extract(self, text, fields):
            release.wait(timeout=3)
            return super().extract(text, fields)

    monkeypatch.setattr("app.main.get_extractor", lambda model: SlowExtractor())
    jobs.clear()
    try:
        with TestClient(app) as client:
            start = client.post(
                "/api/extract/jobs",
                files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
                data={"fields": "Invoice number", "model": NUEXTRACT3_MODEL},
            )
            assert start.status_code == 202
            job_id = start.json()["job_id"]
            assert client.get(f"/api/extract/jobs/{job_id}").json()["status"] == "running"
            rejected = client.post(
                "/api/extract/jobs",
                files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
                data={"fields": "Invoice number"},
            )
            assert rejected.status_code == 429

            release.set()
            deadline = monotonic() + 3
            while monotonic() < deadline:
                polled = client.get(f"/api/extract/jobs/{job_id}").json()
                if polled["status"] == "completed":
                    break
                sleep(0.02)
            assert polled["status"] == "completed"
            assert polled["result"]["results"][0]["value"] == "INV-4421"
            assert polled["result"]["metrics"]["fields_extracted"] == 1
    finally:
        release.set()
        jobs.clear()


def test_job_reports_model_failure_and_unknown_job(monkeypatch):
    class FailingExtractor:
        name = NUEXTRACT3_MODEL

        def extract(self, text, fields):
            raise LocalModelError("out of memory")

    monkeypatch.setattr("app.main.get_extractor", lambda model: FailingExtractor())
    jobs.clear()
    try:
        with TestClient(app) as client:
            start = client.post(
                "/api/extract/jobs",
                files={"pdf": ("invoice.pdf", text_pdf("Invoice number: INV-4421"), "application/pdf")},
                data={"fields": "Invoice number"},
            )
            job_id = start.json()["job_id"]
            deadline = monotonic() + 3
            while monotonic() < deadline:
                polled = client.get(f"/api/extract/jobs/{job_id}").json()
                if polled["status"] == "failed":
                    break
                sleep(0.02)
            assert polled == {"status": "failed", "detail": "out of memory"}
            assert client.get("/api/extract/jobs/missing").status_code == 404
    finally:
        jobs.clear()
