"""Optional PP-StructureV3 parser for the v2 benchmark, never the default path."""

from functools import lru_cache

from app.v2_document import Block, ParseError


@lru_cache(maxsize=1)
def _pipeline():
    try:
        from paddleocr import PPStructureV3
        return PPStructureV3(
            device="cpu", cpu_threads=4,
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            layout_detection_model_name="PP-DocLayout-M",
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, use_seal_recognition=False,
            use_formula_recognition=False, use_chart_recognition=False,
            use_region_detection=False, use_table_recognition=True,
            format_block_content=True,
        )
    except Exception as exc:
        raise ParseError(f"Could not initialize PP-StructureV3: {exc}") from exc


def _bbox(value) -> tuple[float, float, float, float]:
    values = value.tolist() if hasattr(value, "tolist") else value
    if len(values) == 4 and not isinstance(values[0], (list, tuple)):
        return tuple(float(number) for number in values)
    points = values
    return (float(min(point[0] for point in points)), float(min(point[1] for point in points)),
            float(max(point[0] for point in points)), float(max(point[1] for point in points)))


def _table_blocks(text: str, box: tuple[float, float, float, float]) -> list[Block]:
    rows = []
    if "<table" in text.lower():
        from bs4 import BeautifulSoup

        table = BeautifulSoup(text, "html.parser").find("table")
        if table is not None:
            for row in table.find_all("tr"):
                cells = tuple(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"], recursive=False))
                if any(cells):
                    rows.append(Block(" | ".join(cells), box, "table_row", cells, "ppstructure"))
            if rows:
                return rows
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if cells and any(cells) and not all(set(cell) <= {"-", ":", " "} for cell in cells):
            rows.append(Block(" | ".join(cells), box, "table_row", cells, "ppstructure"))
    return rows or [Block(text, box, "table", source="ppstructure")]


def parse_pages(pdf_bytes: bytes, page_numbers: list[int]) -> dict[int, list[Block]]:
    import numpy as np
    import pymupdf

    pipeline = _pipeline()
    parsed = {}
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            for number in page_numbers:
                pixmap = document[number - 1].get_pixmap(dpi=200, alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n)[:, :, ::-1].copy()
                result = next(iter(pipeline.predict(input=image)), None)
                payload = getattr(result, "json", result) if result is not None else {}
                if callable(payload):
                    payload = payload()
                payload = payload.get("res", payload)
                blocks = []
                for item in payload.get("parsing_res_list", []):
                    text = str(item.get("block_content") or "").strip()
                    if not text:
                        continue
                    box = tuple(value * 72 / 200 for value in _bbox(item["block_bbox"]))
                    if item.get("block_label") == "table":
                        blocks.extend(_table_blocks(text, box))
                    else:
                        blocks.append(Block(text, box, source="ppstructure"))
                if not blocks:
                    ocr = payload.get("overall_ocr_res") or {}
                    for text, polygon in zip(ocr.get("rec_texts", []), ocr.get("rec_polys", [])):
                        if str(text).strip():
                            box = tuple(value * 72 / 200 for value in _bbox(polygon))
                            blocks.append(Block(str(text).strip(), box, source="ppstructure"))
                parsed[number] = blocks
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"PP-StructureV3 failed: {exc}") from exc
    return parsed
