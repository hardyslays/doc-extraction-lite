import json
import asyncio

import httpx
import pymupdf
import pytest

from app.main import app
from app.v2_document import Block, Document, Page, ParseError, parse_pdf
from app.v2_extract import ExtractionError, extract_structured
from app.v2_pipeline import run_v2_pipeline
from app.v2_schema import blank_data, parse_template
from app.v2_worker import V2Worker, WorkerError


def pdf_with_text(text: str, *, rotate: bool = False) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    if rotate:
        page.set_rotation(90)
    output = document.tobytes()
    document.close()
    return output


def mixed_pdf() -> bytes:
    source = pymupdf.open(stream=pdf_with_text("Scanned invoice INV-4421"), filetype="pdf")
    image = source[0].get_pixmap(dpi=120).tobytes("png")
    source.close()
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Native invoice number INV-4421")
    document.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=image)
    output = document.tobytes()
    document.close()
    return output


def test_template_accepts_nested_fields_and_requires_one_input():
    template = parse_template('{"invoice": {"date": "date"}, "items": [{"amount": "number"}]}', "")
    assert blank_data(template) == {"invoice": {"date": None}, "items": None}
    assert parse_template("", "Invoice number\nInvoice number") == {"Invoice number": "verbatim-string"}
    with pytest.raises(ValueError, match="exactly one"):
        parse_template("{}", "A")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_template('{"x": "made-up-type"}', "")
    with pytest.raises(ValueError, match="JSON object"):
        parse_template('"date"', "")


def test_native_and_rotated_page_keep_text_without_ocr(monkeypatch):
    monkeypatch.setattr("app.v2_document._docling_pages", lambda *_: pytest.fail("OCR used for native page"))
    for rotate in (False, True):
        document = parse_pdf(pdf_with_text("Invoice number INV-4421", rotate=rotate))
        assert document.pages[0].route == "native"
        assert "INV-4421" in document.pages[0].text
        assert document.pages[0].blocks[0].bbox


def test_scanned_and_mixed_pages_route_independently(monkeypatch):
    called = []

    def fake_docling(_content, page_numbers):
        called.append(page_numbers)
        return {number: [Block("Scanned invoice INV-4421", (0, 0, 300, 100))] for number in page_numbers}

    monkeypatch.setattr("app.v2_document._docling_pages", fake_docling)
    document = parse_pdf(mixed_pdf())
    assert [page.route for page in document.pages] == ["native", "ocr"]
    assert called == [[2]]
    assert "Scanned invoice" in document.pages[1].text


def test_mixed_content_page_keeps_native_and_ocr_blocks(monkeypatch):
    source = pymupdf.open(stream=pdf_with_text("Pencils 2 99.00"), filetype="pdf")
    image = source[0].get_pixmap(dpi=120).tobytes("png")
    source.close()
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Invoice Number: INV-4421")
    page.insert_image(pymupdf.Rect(0, 150, 595, 430), stream=image, keep_proportion=False)
    pdf = document.tobytes()
    document.close()
    monkeypatch.setattr("app.v2_document._docling_pages", lambda *_: {
        1: [Block("Pencils 2 99.00", (72, 160, 300, 190), source="docling")]})
    parsed = parse_pdf(pdf)
    assert parsed.pages[0].route == "mixed"
    assert {block.source for block in parsed.pages[0].blocks} == {"native", "docling"}
    assert "INV-4421" in parsed.pages[0].text
    assert "Pencils" in parsed.pages[0].text


def test_blank_and_malformed_pdf_are_distinct():
    document = pymupdf.open()
    document.new_page()
    blank = document.tobytes()
    document.close()
    assert parse_pdf(blank).pages[0].route == "empty"
    with pytest.raises(ParseError, match="Unable to inspect PDF"):
        parse_pdf(b"not a pdf")


def test_ocr_with_no_text_is_incomplete(monkeypatch):
    monkeypatch.setattr("app.v2_document._docling_pages", lambda *_: {2: []})
    document = parse_pdf(mixed_pdf())
    assert [page.route for page in document.pages] == ["native", "ocr_failed"]
    monkeypatch.setattr("app.v2_pipeline.parse_pdf", lambda *_args, **_kwargs: document)
    monkeypatch.setattr("app.v2_pipeline.load_extractor", lambda *_: None)
    monkeypatch.setattr("app.v2_pipeline.extract_structured",
                        lambda *_: ({"number": None}, {}, 0, []))
    result = run_v2_pipeline(b"pdf", {"number": "verbatim-string"}, "fake")
    assert result["complete"] is False
    assert result["metrics"]["page_routes"]["2"] == "ocr_failed"
    assert "page(s): 2" in result["warnings"][0]


class FakeExtractor:
    name = "fake"

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    _processor = Tokenizer()

    def __init__(self, response):
        self.response = response

    def _generate(self, text, template):
        return json.dumps(self.response(text))


def test_nested_values_have_page_evidence_and_missing_leaves(monkeypatch):
    source = "Invoice number: INV-4421\nDate: October 18, 2026\nTotal: $99.00\nDescription | Quantity | Amount\nPencils | 2 | 99.00"
    document = Document([Page(1, "native", [Block(source, (0, 0, 200, 200))])], 0, 0)
    template = parse_template('{"invoice": {"number": "verbatim-string", "date": "date", "total": "number"}, "items": [{"description": "verbatim-string", "quantity": "integer", "tax": "number"}]}', "")
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: FakeExtractor(lambda text: {
        "invoice": {"number": "INV-4421", "date": "2026-10-18", "total": 99},
        "items": [{"description": "Pencils", "quantity": 2, "tax": 1000}],
    }))
    data, evidence, calls, warnings = extract_structured(document, template, "numind/NuExtract-1.5-tiny")
    assert data == {"invoice": {"number": "INV-4421", "date": "2026-10-18", "total": 99.0},
                    "items": [{"description": "Pencils", "quantity": 2, "tax": None}]}
    assert evidence["/invoice/date"] == {"quote": "October 18, 2026", "page": 1}
    assert evidence["/items/0/description"] == {"quote": "Pencils", "page": 1}
    assert "/items/0/tax" not in evidence
    assert calls == 1 and warnings == []


def test_ambiguous_date_and_conflicting_values_abstain(monkeypatch):
    document = Document([
        Page(1, "native", [Block("Date: 10/11/2026 Invoice: A-11", (0, 0, 200, 20))]),
        Page(2, "native", [Block("Invoice: B-22", (0, 0, 200, 20))]),
    ], 0, 0)
    template = parse_template('{"date": "date", "invoice": "verbatim-string"}', "")
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: FakeExtractor(
        lambda text: {"date": "2026-10-11", "invoice": "A-11" if "A-11" in text else "B-22"}))
    data, evidence, calls, warnings = extract_structured(document, template, "numind/NuExtract-1.5-tiny")
    assert data == {"date": None, "invoice": None}
    assert evidence == {}
    assert calls == 2
    assert any("/invoice" in warning for warning in warnings)


def test_invalid_model_json_is_reported(monkeypatch):
    extractor = FakeExtractor(lambda _: {})
    extractor._generate = lambda *_: "not-json"
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: extractor)
    document = Document([Page(1, "native", [Block("Invoice: A-11", (0, 0, 200, 20))])], 0, 0)
    with pytest.raises(ExtractionError, match="invalid JSON"):
        extract_structured(document, {"invoice": "verbatim-string"}, "numind/NuExtract-1.5-tiny")


def test_date_time_requires_quoted_time(monkeypatch):
    document = Document([Page(1, "native", [Block("Issued: 2026-10-18 14:30:00Z", (0, 0, 200, 20))])], 0, 0)
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: FakeExtractor(
        lambda text: {"issued_at": "2026-10-18T14:30:00+00:00"}))
    data, evidence, _, _ = extract_structured(document, {"issued_at": "date-time"}, "fake")
    assert data == {"issued_at": "2026-10-18T14:30:00+00:00"}
    assert evidence["/issued_at"] == {"quote": "2026-10-18 14:30:00Z", "page": 1}


def test_line_item_cannot_combine_values_from_different_rows(monkeypatch):
    document = Document([Page(1, "native", [Block(
        "Description | Qty\nPencils | 2\nPens | 3", (0, 0, 200, 60))])], 0, 0)
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: FakeExtractor(
        lambda text: {"items": [{"description": "Pencils", "quantity": 3}]}))
    data, evidence, _, _ = extract_structured(document,
                                               {"items": [{"description": "verbatim-string",
                                                           "quantity": "integer"}]}, "fake")
    assert data == {"items": None}
    assert evidence == {}


def test_unresolved_field_gets_bounded_followup(monkeypatch):
    pages = [Page(number, "native", [Block("Invoice pending", (0, 0, 200, 20))])
             for number in range(1, 9)]
    pages.append(Page(9, "native", [Block("INV-4421", (0, 0, 200, 20))]))
    document = Document(pages, 0, 0)
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: FakeExtractor(
        lambda text: {"invoice": "INV-4421" if "INV-4421" in text else None}))
    data, evidence, calls, warnings = extract_structured(document,
                                                          {"invoice": "verbatim-string"}, "fake")
    assert data == {"invoice": "INV-4421"}
    assert evidence["/invoice"]["page"] == 9
    assert calls == 9
    assert warnings == ["Only 8 of 9 document chunks were fully examined."]


def test_identical_line_items_in_one_chunk_are_preserved(monkeypatch):
    document = Document([Page(1, "native", [Block(
        "Pencils | 2\nPencils | 2", (0, 0, 200, 40))])], 0, 0)
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: FakeExtractor(
        lambda text: {"items": [{"description": "Pencils", "quantity": 2},
                                {"description": "Pencils", "quantity": 2}]}))
    data, evidence, _, _ = extract_structured(document,
                                               {"items": [{"description": "verbatim-string",
                                                           "quantity": "integer"}]}, "fake")
    assert len(data["items"]) == 2
    assert set(evidence) == {"/items/0/description", "/items/0/quantity",
                             "/items/1/description", "/items/1/quantity"}


def test_duplicate_model_items_need_distinct_source_rows(monkeypatch):
    document = Document([Page(1, "native", [Block("Pencils | 2", (0, 0, 200, 20))])], 0, 0)
    monkeypatch.setattr("app.v2_extract.load_extractor", lambda _: FakeExtractor(
        lambda text: {"items": [{"description": "Pencils", "quantity": 2},
                                {"description": "Pencils", "quantity": 2}]}))
    data, _, _, _ = extract_structured(document,
                                       {"items": [{"description": "verbatim-string",
                                                   "quantity": "integer"}]}, "fake")
    assert len(data["items"]) == 1


def test_ppstructure_adapter_uses_original_page_numbers_and_pdf_boxes(monkeypatch):
    from app import v2_ppstructure

    class Result:
        json = {"res": {"parsing_res_list": [
            {"block_label": "table", "block_content": "| Description | Qty |\n| Pencils | 2 |",
             "block_bbox": [200, 300, 600, 400]}]}}

    class Pipeline:
        def predict(self, input):
            return iter([Result()])

    monkeypatch.setattr(v2_ppstructure, "_pipeline", lambda: Pipeline())
    parsed = v2_ppstructure.parse_pages(mixed_pdf(), [2])
    assert parsed[2][1].cells == ("Pencils", "2")
    assert parsed[2][1].bbox == (72.0, 108.0, 216.0, 144.0)


def test_ppstructure_html_table_keeps_rows_and_cells():
    from app.v2_ppstructure import _table_blocks

    blocks = _table_blocks("<table><tr><th>Description</th><th>Qty</th></tr>"
                           "<tr><td>Pencils</td><td>2</td></tr></table>", (0, 0, 100, 100))
    assert [block.cells for block in blocks] == [("Description", "Qty"), ("Pencils", "2")]


def test_v2_api_accepts_template_and_keeps_v1_route(monkeypatch):
    from app import v2_api

    expected = {"model": "fake", "pages_processed": 1, "complete": True,
                "data": {"invoice": "INV-4421"},
                "evidence": {"/invoice": {"quote": "INV-4421", "page": 1}},
                "warnings": [], "metrics": {"total_seconds": 0.1, "model_calls": 1}}
    monkeypatch.setattr(v2_api.worker, "run", lambda *_: expected)
    async def immediate(*args):
        return v2_api.worker.run(*args)
    monkeypatch.setattr(v2_api, "run_worker", immediate)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v2/extract", files={"pdf": ("x.pdf", pdf_with_text("Invoice INV-4421"), "application/pdf")},
                                         data={"template": '{"invoice":"verbatim-string"}'})
            assert response.status_code == 200
            assert response.json()["data"] == expected["data"]
            invalid = await client.post("/api/v2/extract", files={"pdf": ("x.pdf", b"abc", "application/pdf")},
                                        data={"template": '{"invoice":"verbatim-string"}', "fields": "Invoice"})
            assert invalid.status_code == 422
    asyncio.run(check())
    assert "/api/extract" in app.openapi()["paths"]


def test_v2_job_reports_incomplete_processing(monkeypatch):
    from app import v2_api

    async def immediate(*args):
        return {"model": "fake", "parser": "docling", "pages_processed": 1,
                "complete": False, "data": {"invoice": None}, "evidence": {},
                "warnings": ["Structured OCR returned no text for page(s): 1"], "metrics": {}}

    monkeypatch.setattr(v2_api, "run_worker", immediate)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v2/extract/jobs",
                                         files={"pdf": ("x.pdf", pdf_with_text("Invoice"), "application/pdf")},
                                         data={"fields": "invoice"})
            assert response.status_code == 202
            await asyncio.sleep(0)
            job = await client.get("/api/v2/extract/jobs/" + response.json()["job_id"])
            assert job.json()["status"] == "incomplete"
            assert job.json()["result"]["data"] == {"invoice": None}
    asyncio.run(check())


def test_worker_timeout_restarts_child(monkeypatch):
    instance = V2Worker()
    class Connection:
        def send(self, _):
            pass
        def poll(self, _):
            return False
    closed = []
    monkeypatch.setattr(instance, "_start", lambda: setattr(instance, "_connection", Connection()))
    monkeypatch.setattr(instance, "close", lambda: closed.append(True))
    with pytest.raises(WorkerError, match="job limit"):
        instance.run(b"pdf", {"x": "string"}, "numind/NuExtract-1.5-tiny", False)
    assert closed == [True]


def test_worker_crash_restarts_child(monkeypatch):
    instance = V2Worker()

    class Connection:
        def send(self, _):
            pass
        def poll(self, _):
            return True
        def recv(self):
            raise EOFError

    closed = []
    monkeypatch.setattr(instance, "_start", lambda: setattr(instance, "_connection", Connection()))
    monkeypatch.setattr(instance, "close", lambda: closed.append(True))
    with pytest.raises(WorkerError, match="stopped unexpectedly"):
        instance.run(b"pdf", {"x": "string"}, "numind/NuExtract-1.5-tiny", False)
    assert closed == [True]
