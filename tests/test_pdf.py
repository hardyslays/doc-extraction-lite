from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import app.pdf as pdf


def pdf_with_blank_page() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def pdf_with_text(text: str) -> bytes:
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
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extract_pages_uses_ocr_for_pages_without_embedded_text(monkeypatch):
    uploaded_content = pdf_with_blank_page()

    def fake_ocr(content, page_indexes):
        assert content == uploaded_content
        assert list(page_indexes) == [0]
        return {0: "Scanned invoice INV-4421"}

    monkeypatch.setattr(pdf, "extract_ocr_pages", fake_ocr)

    assert pdf.extract_pages(uploaded_content) == ["Scanned invoice INV-4421"]


def test_extract_pages_keeps_embedded_text_without_initializing_ocr(monkeypatch):
    def fail_if_called(*_):
        raise AssertionError("OCR called")

    monkeypatch.setattr(pdf, "extract_ocr_pages", fail_if_called)

    assert pdf.extract_pages(pdf_with_text("Selectable text")) == ["Selectable text"]


def test_extract_pages_force_ocr_replaces_embedded_text(monkeypatch):
    def fake_ocr(_, page_indexes):
        assert list(page_indexes) == [0]
        return {0: "OCR text"}

    monkeypatch.setattr(pdf, "extract_ocr_pages", fake_ocr)
    assert pdf.extract_pages(pdf_with_text("Selectable text"), force_ocr=True) == ["OCR text"]
