# Synthetic v2 benchmark (2026-10-19)

The JSON files here were generated with `scripts/benchmark_v2.py` on the local
12-thread, 32 GB CPU machine. Each combination had one discarded native-PDF
warm-up followed by one measured run per fixture. PDF/model content is not
recorded in these files. The corpus is reproducible with the script.

- `synthetic-model-comparison.json` covers v1 tiny and v2 Docling with tiny and
  smol on native, scanned, mixed, hybrid-content, table, scanned-table, rotated,
  empty, and malformed fixtures. V1 supports only the three shared scalar
  fields; `shared_precision` and `shared_recall` are the comparable scores.
- `synthetic-parser-comparison.json` covers Docling and PP-StructureV3 with tiny
  on native and scanned-table fixtures. Both adapters recognized the synthetic
  table rows after the PP-Structure HTML table fix.

The samples are too small for a reliable p95 estimate or a claim that v2 is
more accurate than v1 on business documents. The full model run had identical
supported-value accuracy for tiny and smol. Tiny was faster and used less
worker memory, so it remains the default. The parser comparison did not show
an accuracy gain for PP-StructureV3, so Docling remains the default.

Next acceptance gate: collect reviewed real English business PDFs, run each
combination at least twice after warm-up, and inspect wrong values, OCR/table
errors, p50/p95 latency, and peak worker memory. Keep the default under two
minutes on typical 2–5 page PDFs.
