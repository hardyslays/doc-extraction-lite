"""Generate local synthetic PDFs and benchmark v1/v2 after a discarded warm-up.

Run `python scripts/benchmark_v2.py --out /tmp/doc-v2-fixtures` to write the corpus.
Add `--run` to execute local models; this can take many minutes on CPU.
"""

import argparse
import json
import math
import resource
import statistics
import sys
from pathlib import Path
from time import perf_counter

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.nuextract import SMOL_MODEL, TINY_MODEL
from app.v2_document import parse_pdf
from app.v2_worker import V2Worker, WorkerError


TEMPLATE = {
    "invoice": {"number": "verbatim-string", "date": "date", "total": "number"},
    "items": [{"description": "verbatim-string", "quantity": "integer", "amount": "number"}],
}
EXPECTED = {"invoice": {"number": "INV-4421", "date": "2026-10-18", "total": 99.0},
            "items": [{"description": "Pencils", "quantity": 2, "amount": 99.0}]}
EVIDENCE = {"/invoice/number": {"quote": "INV-4421", "page": 1},
            "/invoice/date": {"quote": "October 18, 2026", "page": 1},
            "/invoice/total": {"quote": "99.00", "page": 1},
            "/items/0/description": {"quote": "Pencils", "page": 2},
            "/items/0/quantity": {"quote": "2", "page": 2},
            "/items/0/amount": {"quote": "99.00", "page": 2}}


def _invoice_document(*, table: bool = False, rotated: bool = False) -> bytes:
    document = pymupdf.open()
    first = document.new_page()
    for y, line in enumerate(("Invoice Number: INV-4421", "Invoice Date: October 18, 2026",
                              "Total: 99.00"), 1):
        first.insert_text((72, 72 + y * 30), line, fontsize=12)
    if rotated:
        first.set_rotation(90)
    second = document.new_page()
    if table:
        xs, ys = (72, 290, 390, 500), (72, 105, 138)
        for x in xs:
            second.draw_line((x, ys[0]), (x, ys[-1]))
        for y in ys:
            second.draw_line((xs[0], y), (xs[-1], y))
        for y, cells in ((94, ("Description", "Qty", "Amount")),
                         (127, ("Pencils", "2", "99.00"))):
            for x, cell in zip((80, 300, 400), cells):
                second.insert_text((x, y), cell, fontsize=11)
    else:
        second.insert_text((72, 90), "Description | Qty | Amount", fontsize=12)
        second.insert_text((72, 120), "Pencils | 2 | 99.00", fontsize=12)
    output = document.tobytes()
    document.close()
    return output


def _rasterize(pdf_bytes: bytes, pages: set[int]) -> bytes:
    source = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    target = pymupdf.open()
    for number, page in enumerate(source, 1):
        if number in pages:
            image = page.get_pixmap(dpi=170, alpha=False).tobytes("png")
            target.new_page(width=page.rect.width, height=page.rect.height).insert_image(
                page.rect, stream=image)
        else:
            target.insert_pdf(source, from_page=number - 1, to_page=number - 1)
    output = target.tobytes()
    source.close()
    target.close()
    return output


def _hybrid_page(pdf_bytes: bytes) -> bytes:
    source = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    image = source[1].get_pixmap(dpi=170, alpha=False,
                                 clip=pymupdf.Rect(0, 0, 595, 150)).tobytes("png")
    target = pymupdf.open()
    target.insert_pdf(source, from_page=0, to_page=0)
    page = target.new_page()
    page.insert_text((72, 90), "Description | Qty | Amount", fontsize=12)
    page.insert_image(pymupdf.Rect(0, 150, 595, 300), stream=image)
    output = target.tobytes()
    source.close()
    target.close()
    return output


def fixtures() -> dict[str, dict]:
    native = _invoice_document()
    table = _invoice_document(table=True)
    empty = pymupdf.open()
    empty.new_page()
    empty.new_page()
    blank = empty.tobytes()
    empty.close()
    expected = lambda: json.loads(json.dumps(EXPECTED))
    evidence = lambda: json.loads(json.dumps(EVIDENCE))
    return {
        "native": {"pdf": native, "expected": expected(), "evidence": evidence()},
        "scanned": {"pdf": _rasterize(native, {1, 2}), "expected": expected(), "evidence": evidence()},
        "mixed": {"pdf": _rasterize(native, {2}), "expected": expected(), "evidence": evidence()},
        "hybrid_page": {"pdf": _hybrid_page(native), "expected": expected(), "evidence": evidence()},
        "table": {"pdf": table, "expected": expected(), "evidence": evidence()},
        "scanned_table": {"pdf": _rasterize(table, {1, 2}),
                          "expected": expected(), "evidence": evidence()},
        "rotated": {"pdf": _invoice_document(rotated=True), "expected": expected(), "evidence": evidence()},
        "empty": {"pdf": blank, "expected": {"invoice": {"number": None, "date": None,
                                                         "total": None}, "items": None}, "evidence": {}},
        "malformed": {"pdf": b"not a PDF", "error": "parse"},
    }


def _flatten(value, path="") -> dict[str, object]:
    if isinstance(value, dict):
        return {key: leaf for name, item in value.items()
                for key, leaf in _flatten(item, path + "/" + name).items()}
    if isinstance(value, list):
        return {key: leaf for index, item in enumerate(value)
                for key, leaf in _flatten(item, path + "/" + str(index)).items()}
    return {path: value}


def _score(actual: dict, expected: dict, evidence: dict, expected_evidence: dict) -> dict:
    actual_values = _flatten(actual)
    expected_values = _flatten(expected)
    correct = wrong = missed = evidence_errors = 0
    supported = sum(value is not None for value in expected_values.values())
    for path in set(actual_values) | set(expected_values):
        got, wanted = actual_values.get(path), expected_values.get(path)
        if got is None:
            missed += wanted is not None
        elif got == wanted:
            correct += 1
            if path in expected_evidence and evidence.get(path) != expected_evidence[path]:
                evidence_errors += 1
        else:
            wrong += 1
    return {"correct": correct, "wrong": wrong, "missed": missed,
            "supported": supported, "evidence_errors": evidence_errors}


def _percentile(values: list[float], percent: float) -> float | None:
    return sorted(values)[max(0, math.ceil(len(values) * percent) - 1)] if values else None


def _summarize(records: list[dict]) -> dict:
    scored = [record["score"] for record in records if "score" in record]
    totals = {key: sum(score[key] for score in scored) for key in
              ("correct", "wrong", "missed", "supported", "evidence_errors")}
    predicted = totals["correct"] + totals["wrong"]
    durations = [record["seconds"] for record in records if "seconds" in record]
    memories = [record["peak_rss_mb"] for record in records if record.get("peak_rss_mb") is not None]
    shared = [record["shared_score"] for record in records if "shared_score" in record]
    shared_correct = sum(score["correct"] for score in shared)
    shared_wrong = sum(score["wrong"] for score in shared)
    shared_supported = sum(score["supported"] for score in shared)
    return {"precision": totals["correct"] / predicted if predicted else None,
            "recall": totals["correct"] / totals["supported"] if totals["supported"] else None,
            "false_answer_rate": totals["wrong"] / predicted if predicted else None,
            "shared_precision": shared_correct / (shared_correct + shared_wrong) if shared_correct + shared_wrong else None,
            "shared_recall": shared_correct / shared_supported if shared_supported else None,
            "evidence_errors": totals["evidence_errors"],
            "p50_seconds": statistics.median(durations) if durations else None,
            "p95_seconds": _percentile(durations, .95),
            "peak_rss_mb": max(memories) if memories else None,
            "failures": sum("error" in record for record in records)}


def _v1_result(pdf_bytes: bytes, model: str) -> dict:
    from app.main import run_pipeline

    result = run_pipeline(pdf_bytes, ["Invoice Number", "Invoice Date", "Total"], model,
                          False, perf_counter())
    values = {item.field: item.value for item in result.results}
    return {"invoice": {"number": values.get("Invoice Number"),
                        "date": values.get("Invoice Date"), "total": values.get("Total")},
            "items": None}


def run_benchmark(corpus: dict, models: list[str], parsers: list[str], repeats: int,
                  include_v1: bool) -> dict:
    groups = {}
    worker = V2Worker()
    try:
        combinations = [("v2/" + parser, model) for parser in parsers for model in models]
        if include_v1:
            combinations = [("v1", model) for model in models if model == TINY_MODEL] + combinations
        for route, model in combinations:
            records = []
            # A discarded native run warms model weights independently for each combination.
            try:
                if route == "v1":
                    _v1_result(corpus["native"]["pdf"], model)
                else:
                    worker.run(corpus["native"]["pdf"], TEMPLATE, model, False, route[3:])
            except Exception as exc:
                groups[route + "/" + model] = {"error": f"Warm-up failed: {type(exc).__name__}"}
                worker.close()
                worker = V2Worker()
                continue
            for _ in range(repeats):
                for name, fixture in corpus.items():
                    started = perf_counter()
                    try:
                        if route == "v1":
                            data = _v1_result(fixture["pdf"], model)
                            evidence = {}
                            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                            expected = {"invoice": fixture.get("expected", {}).get("invoice"),
                                        "items": None}
                            if expected["invoice"]:
                                expected["invoice"] = {**expected["invoice"],
                                                       "date": fixture["evidence"]["/invoice/date"]["quote"],
                                                       "total": "99.00"}
                            expected_evidence = {}
                        else:
                            result = worker.run(fixture["pdf"], TEMPLATE, model, False, route[3:])
                            data, evidence = result["data"], result["evidence"]
                            peak = result["metrics"]["peak_rss_mb"]
                            expected, expected_evidence = fixture.get("expected"), fixture.get("evidence", {})
                        if fixture.get("error"):
                            records.append({"fixture": name, "error": "Expected parsing failure but succeeded"})
                            continue
                        record = {"fixture": name, "seconds": perf_counter() - started,
                                  "peak_rss_mb": peak,
                                  "score": _score(data, expected, evidence, expected_evidence)}
                        record["shared_score"] = _score(
                            {"invoice": data["invoice"]}, {"invoice": expected["invoice"]},
                            evidence, {key: value for key, value in expected_evidence.items()
                                       if key.startswith("/invoice/")})
                        if route != "v1":
                            parsed = parse_pdf(fixture["pdf"], parser=route[3:])
                            record["ocr_errors"] = sum(source["quote"] not in parsed.pages[source["page"] - 1].text
                                                       for source in expected_evidence.values())
                            record["table_errors"] = int(name in {"table", "scanned_table"} and not any(
                                block.kind == "table_row" and "Pencils" in block.cells
                                for page in parsed.pages for block in page.blocks))
                            record["complete"] = result["complete"]
                        records.append(record)
                    except Exception as exc:
                        if fixture.get("error"):
                            records.append({"fixture": name, "expected_failure": type(exc).__name__})
                        else:
                            records.append({"fixture": name, "error": type(exc).__name__})
            groups[route + "/" + model] = {"summary": _summarize(records), "records": records}
            worker.close()
            worker = V2Worker()
    finally:
        worker.close()
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/doc-v2-fixtures"))
    parser.add_argument("--run", action="store_true", help="run local model inference after writing fixtures")
    parser.add_argument("--models", nargs="+", default=[TINY_MODEL, SMOL_MODEL])
    parser.add_argument("--parsers", nargs="*", default=["docling", "ppstructure"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--include-v1", action="store_true")
    parser.add_argument("--fixtures", nargs="+", help="measure a subset (native warm-up is always available)")
    args = parser.parse_args()
    corpus = fixtures()
    if args.fixtures:
        names = ["native", *args.fixtures]
        unknown = set(names) - set(corpus)
        if unknown:
            parser.error("unknown fixtures: " + ", ".join(sorted(unknown)))
        corpus = {name: fixture for name, fixture in corpus.items() if name in names}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, fixture in corpus.items():
        (args.out / f"{name}.pdf").write_bytes(fixture["pdf"])
    manifest = {name: {key: value for key, value in fixture.items() if key != "pdf"}
                for name, fixture in corpus.items()}
    (args.out / "expected.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(corpus)} fixtures to {args.out}")
    if args.run:
        results = run_benchmark(corpus, args.models, args.parsers, args.repeats, args.include_v1)
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({key: value.get("summary", value) for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
