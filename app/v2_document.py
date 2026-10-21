"""Page-aware PDF parsing shared by native text and structured OCR paths."""

from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from time import perf_counter


MAX_RENDER_PIXELS = 12_000_000


@dataclass(frozen=True)
class Block:
    text: str
    bbox: tuple[float, float, float, float]
    kind: str = "text"
    cells: tuple[str, ...] = ()
    source: str = "native"
    cell_boxes: tuple[tuple[float, float, float, float] | None, ...] = ()


@dataclass
class Page:
    number: int
    route: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text)


@dataclass
class Document:
    pages: list[Page]
    native_seconds: float
    ocr_seconds: float


class ParseError(RuntimeError):
    pass


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return x * y / area


def _native_page(page, number: int, force_ocr: bool) -> Page:
    raw_blocks = page.get_text("blocks", sort=True)
    text_blocks = [Block(str(item[4]).strip(), tuple(item[:4])) for item in raw_blocks
                   if item[6] == 0 and str(item[4]).strip()]
    text = " ".join(block.text for block in text_blocks)
    image_area = sum((box[2] - box[0]) * (box[3] - box[1])
                     for image in page.get_image_info() for box in [image["bbox"]])
    page_area = max(1.0, page.rect.width * page.rect.height)
    usable = sum(char.isalnum() for char in text) >= 20 and text.count("\ufffd") <= max(1, len(text) // 20)
    if force_ocr:
        return Page(number, "ocr")
    if not usable:
        if image_area > 0.05 * page_area or (text.strip() and text.count("\ufffd") > max(1, len(text) // 20)):
            return Page(number, "ocr")
        return Page(number, "native" if text.strip() else "empty", text_blocks)
    if image_area > 0.15 * page_area:
        return Page(number, "mixed", text_blocks)

    tables: list[tuple[tuple[float, float, float, float], list[Block]]] = []
    try:
        for table in page.find_tables().tables:
            rows = table.extract()
            row_blocks = []
            for index, row in enumerate(rows):
                cells = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cells):
                    boxes = tuple(tuple(box) if box else None for box in table.rows[index].cells)
                    row_blocks.append(Block(" | ".join(cells), tuple(table.bbox), "table_row",
                                            tuple(cells), "native", boxes))
            if row_blocks:
                tables.append((tuple(table.bbox), row_blocks))
    except Exception:
        # Text extraction is still useful if table detection does not support this page.
        pass
    blocks = [block for block in text_blocks if not any(_overlap(block.bbox, box) > 0.5 for box, _ in tables)]
    for box, rows in tables:
        blocks.extend(rows)
    blocks.sort(key=lambda block: (block.bbox[1], block.bbox[0]))
    return Page(number, "native", blocks)


@lru_cache(maxsize=1)
def _docling_converter():
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ParseError("Structured OCR is unavailable. Install the v2 dependencies with uv sync --extra v2.") from exc

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True  # The table model runs only on detected table regions.
    options.ocr_options = RapidOcrOptions(lang=["en"], backend="onnxruntime", model_size="small", mode="full_page", scale=200 / 72)
    options.accelerator_options.num_threads = 4
    options.accelerator_options.device = "cpu"
    options.ocr_batch_size = 2
    options.layout_batch_size = 2
    options.table_batch_size = 2
    options.generate_page_images = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def _docling_pages(pdf_bytes: bytes, page_numbers: list[int]) -> dict[int, list[Block]]:
    converter = _docling_converter()
    from docling.datamodel.base_models import DocumentStream
    parsed: dict[int, list[Block]] = {number: [] for number in page_numbers}
    # Convert contiguous page runs together so Docling can batch stages without OCRing native pages.
    runs: list[list[int]] = []
    for number in page_numbers:
        if not runs or number != runs[-1][-1] + 1:
            runs.append([number])
        else:
            runs[-1].append(number)
    for run in runs:
        source = DocumentStream(name="upload.pdf", stream=BytesIO(pdf_bytes))
        result = converter.convert(source, page_range=(run[0], run[-1]), raises_on_error=True)
        doc = result.document
        if doc is None:
            raise ParseError(f"Structured OCR returned no document for pages {run[0]}-{run[-1]}.")
        for item, _level in doc.iterate_items():
            prov = getattr(item, "prov", None) or []
            if not prov or prov[0].page_no not in parsed:
                continue
            number = prov[0].page_no
            bbox = prov[0].bbox.to_top_left_origin(doc.pages[number].size.height)
            box = (bbox.l, bbox.t, bbox.r, bbox.b)
            if hasattr(item, "export_to_dataframe"):
                try:
                    frame = item.export_to_dataframe(doc=doc).fillna("")
                    rows = frame.values.tolist()
                    if all(isinstance(column, str) for column in frame.columns):
                        rows.insert(0, frame.columns.tolist())
                    for row_index, row in enumerate(rows):
                        text = " | ".join(str(cell).strip() for cell in row)
                        if text.strip(" |"):
                            grid = item.data.grid
                            grid_row = grid[row_index] if row_index < len(grid) else []
                            cell_boxes = tuple(
                                tuple((cell.bbox.to_top_left_origin(doc.pages[number].size.height).model_dump()[key]
                                       for key in ("l", "t", "r", "b"))) if cell.bbox else None
                                for cell in grid_row
                            )
                            parsed[number].append(Block(text, box, "table_row",
                                                        tuple(str(cell).strip() for cell in row),
                                                        "docling", cell_boxes))
                    continue
                except Exception as exc:
                    raise ParseError(f"Could not read table on page {number}: {exc}") from exc
            text = str(getattr(item, "text", "") or "").strip()
            if text:
                parsed[number].append(Block(text, box, "text", source="docling"))
    return parsed


def parse_pdf(pdf_bytes: bytes, *, force_ocr: bool = False, parser: str = "docling") -> Document:
    import pymupdf

    started = perf_counter()
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as source:
            if not source.page_count:
                raise ParseError("PDF has no pages.")
            pages = []
            for index, page in enumerate(source):
                if page.rect.width * page.rect.height * (200 / 72) ** 2 > MAX_RENDER_PIXELS:
                    raise ParseError(f"Page {index + 1} exceeds the rendering pixel limit.")
                pages.append(_native_page(page, index + 1, force_ocr))
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"Unable to inspect PDF: {exc}") from exc
    native_seconds = perf_counter() - started
    ocr_numbers = [page.number for page in pages if page.route in {"ocr", "mixed"}]
    if ocr_numbers:
        try:
            if parser == "docling":
                parsed = _docling_pages(pdf_bytes, ocr_numbers)
            elif parser == "ppstructure":
                from app.v2_ppstructure import parse_pages
                parsed = parse_pages(pdf_bytes, ocr_numbers)
            else:
                raise ParseError(f"Unknown structured parser: {parser}")
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Structured OCR failed: {exc}") from exc
        for page in pages:
            if page.route == "mixed":
                original = page.blocks
                recognized = parsed.get(page.number, [])
                page.blocks = original + [block for block in recognized if block.text not in {item.text for item in original}]
                page.blocks.sort(key=lambda block: (block.bbox[1], block.bbox[0]))
                if not recognized:
                    page.route = "mixed_partial"
            elif page.route == "ocr":
                page.blocks = parsed.get(page.number, [])
                if not page.blocks:
                    page.route = "ocr_failed"
    return Document(pages, native_seconds, perf_counter() - started - native_seconds)
