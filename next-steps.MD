# V2: next steps

## 1. Use it

```bash
uv sync --extra dev --extra v2 --extra benchmark
uv run --extra v2 --extra benchmark uvicorn app.main:app --workers 1
```

Open `http://127.0.0.1:8000`. In **V2 structured extraction**, upload a PDF, paste a nested JSON template, and start with **NuExtract-1.5-tiny**. Check the returned value, exact quote, and page for every field. A missing value should be `null`; a failed or incomplete job needs investigation.

## 2. Reproduce the synthetic baseline

```bash
uv run --extra v2 --extra benchmark python scripts/benchmark_v2.py --run --include-v1 --models numind/NuExtract-1.5-tiny numind/NuExtract-1.5-smol --parsers docling --repeats 2 --out /tmp/v2-model-benchmark
uv run --extra v2 --extra benchmark python scripts/benchmark_v2.py --run --models numind/NuExtract-1.5-tiny --parsers docling ppstructure --repeats 2 --out /tmp/v2-parser-benchmark
```

Read each `results.json`; compare **shared** precision/recall for v1 versus v2, then v2's nested fields, false-answer rate, OCR/table errors, p50/p95 time, and peak memory. First model downloads and the discarded warm-up are outside measured runs. The saved one-pass baseline is in [`benchmarks/`](benchmarks/README.md).

## 3. Test the real-document theory

1. Collect 20–30 reviewed, local English business PDFs: native, scanned, mixed, tables, rotations, and documents with missing or conflicting fields.
2. For each PDF, write the expected nested JSON **and** the supporting quote/page for every non-null value. Mark unsupported values `null`.
3. Use v1 with the shared flat fields and v2 with the full nested template; compare tiny and smol. Record wrong answers separately from missing answers, and inspect every wrong quote and OCR/table miss.
4. Extend the runner to score these reviewed PDFs before comparing Docling with PP-StructureV3. Keep **tiny + Docling** unless another combination improves supported-value accuracy **and** keeps warm p95 under **120 seconds** for typical 2–5 page PDFs. Do not claim an accuracy gain from synthetic fixtures alone.

`scripts/benchmark_v2.py` currently scores synthetic PDFs only. The next engineering task for repeatable real-PDF results is to add a reviewed-PDF manifest input to that runner. NuExtract3 is an optional slow CPU check, not the default candidate.
