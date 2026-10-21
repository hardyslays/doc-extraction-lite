from io import BytesIO

from pypdf import PdfReader

from app.ocr import extract_ocr_pages


def extract_pages(pdf_bytes: bytes, force_ocr: bool = False) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    ocr_page_indexes = list(range(len(pages))) if force_ocr else [
        index for index, text in enumerate(pages) if not text
    ]
    if ocr_page_indexes:
        ocr_pages = extract_ocr_pages(pdf_bytes, ocr_page_indexes)
        for page_index, text in ocr_pages.items():
            pages[page_index] = text
    if not any(pages):
        raise ValueError("No text could be extracted from this PDF.")
    return pages
