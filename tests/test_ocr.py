from io import BytesIO

from pypdf import PdfWriter

import app.ocr as ocr


def blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extract_ocr_pages_returns_recognized_lines(monkeypatch):
    class FakeOCR:
        def predict(self, image):
            assert image.shape[2] == 3
            return iter([{"rec_texts": ["Invoice number", "INV-4421", ""]}])

    monkeypatch.setattr(ocr, "get_ocr", lambda: FakeOCR())

    assert ocr.extract_ocr_pages(blank_pdf(), [0]) == {0: "Invoice number\nINV-4421"}
