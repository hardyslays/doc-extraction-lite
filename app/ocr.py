from collections.abc import Iterable
from functools import lru_cache
import os


class LocalOCRError(RuntimeError):
    """Raised when the local PaddleOCR engine cannot process a document."""


@lru_cache(maxsize=1)
def get_ocr():
    """Load the local PP-OCRv6 model only when OCR is first needed."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise LocalOCRError("PaddleOCR is not installed. Run `uv sync` to install OCR support.") from exc

    try:
        return PaddleOCR(
            device="cpu",
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            cpu_threads=min(4, os.cpu_count() or 1),
            text_det_limit_side_len=1536,
        )
    except Exception as exc:
        raise LocalOCRError(f"Unable to initialize the local PP-OCRv6 model: {exc}") from exc


def extract_ocr_pages(pdf_bytes: bytes, page_indexes: Iterable[int]) -> dict[int, str]:
    """Render selected PDF pages and return their PP-OCRv6 text by page index."""
    try:
        import pymupdf
        import numpy as np
    except ImportError as exc:
        raise LocalOCRError("PDF rendering support is not installed. Run `uv sync` to install OCR support.") from exc

    try:
        ocr = get_ocr()
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            page_text: dict[int, str] = {}
            for page_index in page_indexes:
                pixmap = document.load_page(page_index).get_pixmap(dpi=200, alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                # PyMuPDF returns RGB pixels; PaddleOCR's ndarray input uses BGR.
                result = next(iter(ocr.predict(image[:, :, ::-1])), None)
                text_lines = result["rec_texts"] if result is not None else []
                page_text[page_index] = "\n".join(str(text).strip() for text in text_lines if str(text).strip())
            return page_text
    except LocalOCRError:
        raise
    except Exception as exc:
        raise LocalOCRError(f"Unable to OCR this PDF with the local PP-OCRv6 model: {exc}") from exc
