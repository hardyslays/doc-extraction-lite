from io import BytesIO

from pypdf import PdfReader


def extract_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not any(pages):
        raise ValueError("No selectable text was found. This prototype accepts text-native PDFs only.")
    return pages
