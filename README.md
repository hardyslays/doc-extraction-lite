# doc-extraction-lite

Configurable document-extraction service using local NuExtract models.

## Document Field Extractor

A local pipeline for extracting requested fields from PDFs. It runs template-guided NuExtract inference on PDF text (or locally OCRed text), then keeps only values quoted verbatim from a source page.

Choose [`numind/NuExtract-1.5-tiny`](https://huggingface.co/numind/NuExtract-1.5-tiny) (the default, 0.5B parameters) or [`numind/NuExtract3`](https://huggingface.co/numind/NuExtract3) (4B-class, about 5B total parameters). Only the selected checkpoint loads; switching models evicts the previous checkpoint from the in-process cache. First use downloads weights to the local Hugging Face cache. NuExtract3 needs substantially more memory and will be slow on CPU. No document content is sent to a hosted API.

### Run it

1. Install the locked application and CPU-only PyTorch runtime with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync --extra dev --extra v2
   ```

2. Start the web app:

   ```bash
   uv run uvicorn app.main:app --reload
   ```

   Visit `http://127.0.0.1:8000`.

The server starts without downloading model weights. Select a model in the UI, or send its ID in the API's `model` form field; the selected model loads on first use. The first request's timing includes this load/download. The UI submits a background job and polls for the result, so slow NuExtract3 inference does not hold one tunnel request open until it times out. This does not make CPU inference faster.

### OCR for scanned PDFs

Pages without selectable text are rendered locally and passed to the PP-OCRv6 small model before extraction. Check **Force OCR** in the UI, or include `force_ocr=true` in the multipart API request, to render and OCR every page even when a text layer is present. The PaddleOCR weights are downloaded once when OCR is first used and then run locally from the model cache; no document content is sent to a hosted OCR service.

### Current pipeline

```text
PDF upload → PyPDF text per page (or local PP-OCRv6) → NuExtract JSON extraction
          → verbatim evidence validation → first grounded result per field → UI
```

The extractor sits behind a small `FieldExtractor` protocol. Long documents are processed in overlapping text windows, with values mapped back to their PDF page. Missing fields return `null`. NuExtract does not provide calibrated per-field confidence, so `confidence` is `null` for both found and missing values. A value's presence in the text is necessary but does not prove that the model selected the correct field.

### API

Send `POST /api/extract` as multipart form data with a PDF in `pdf`, newline-separated field names in `fields`, and optional `model` (`numind/NuExtract-1.5-tiny` or `numind/NuExtract3`). Optional `force_ocr=true` OCRs all pages. The response includes time spent extracting text, loading the model, and running inference:

```json
{
  "model": "numind/NuExtract-1.5-tiny",
  "pages_processed": 1,
  "metrics": {
    "total_seconds": 3.52,
    "text_extraction_seconds": 0.04,
    "model_load_seconds": 0.01,
    "inference_seconds": 3.47,
    "fields_requested": 1,
    "fields_extracted": 1
  },
  "results": [
    {
      "field": "Invoice number",
      "value": "INV-4421",
      "confidence": null,
      "evidence": "INV-4421",
      "page": 1
    }
  ]
}
```

For long-running requests or access through a tunnel, use `POST /api/extract/jobs` with the same multipart fields. It returns HTTP 202 and `{"job_id": "...", "status": "running"}`. Poll `GET /api/extract/jobs/{job_id}` until `status` is `completed` (with the response above in `result`) or `failed` (with a `detail` message). The synchronous `/api/extract` endpoint remains available for local clients. Jobs are kept only in this server process: a restart loses them, and only one background extraction runs at a time to avoid loading competing large models. Run a single Uvicorn worker when using the in-memory job API.

### Verify

```bash
uv run --extra v2 pytest
node --test tests/test_frontend.mjs
```

### V2 structured extraction

V2 is available in the main UI and through `POST /api/v2/extract` or
`POST /api/v2/extract/jobs`. Send a PDF and **exactly one** of these multipart
form fields:

- `template`: a JSON object with nested objects, one-item arrays for repeated
  records, and leaves of `verbatim-string`, `string`, `date`, `date-time`,
  `number`, `integer`, or `currency`.
- `fields`: newline-separated v1 field names, converted to a flat template.

Optional `model` is `numind/NuExtract-1.5-tiny` (default),
`numind/NuExtract-1.5-smol`, or `numind/NuExtract3`. Optional `force_ocr=true`
routes every page through structured OCR. NuExtract3 may take more than two
minutes on CPU. A first run can download model weights and is excluded from
warm-run timing. The Docling/RapidOCR layout and OCR weights are also needed
for scanned pages; they download on first use when network access is available.

Example template:

```json
{
  "invoice": {"number": "verbatim-string", "date": "date", "total": "number"},
  "items": [{"description": "verbatim-string", "quantity": "integer", "amount": "number"}]
}
```

The result contains `data`, `evidence` keyed by JSON pointer paths such as
`/invoice/number` and `/items/0/amount`, `complete`, `warnings`, and stage
timings in `metrics`. Each evidence entry has an exact source quote and a
one-based PDF page number. Unsupported values and absent collections are
`null`. A failed parse, invalid model output, OCR page with no recognized
text, or skipped chunks are reported as failed or incomplete processing;
these are distinct from a trustworthy `null` result. The synchronous endpoint
returns the result directly. The jobs endpoint returns a job ID to poll at
`GET /api/v2/extract/jobs/{job_id}`; states are `running`, `completed`,
`incomplete`, or `failed`.

V2 uses PyMuPDF blocks and tables on native pages. Scanned or heavily imaged
pages use Docling with RapidOCR English PP-OCRv6 small models on ONNX Runtime.
The optional PP-StructureV3 parser is available in the benchmark runner, not
as the default API path. V2 runs in a reusable isolated process with one job
at a time, a 20 MB upload limit, and a 180-second job limit for tiny and smol
(600 seconds for NuExtract3). Use one Uvicorn server worker with the in-memory
jobs API.

Generate synthetic native, scanned, mixed, hybrid-content, table, scanned-table, rotated, empty, and malformed
fixtures, then benchmark warmed local models:

```bash
uv run --extra v2 --extra benchmark python scripts/benchmark_v2.py --out /tmp/doc-v2-fixtures
uv run --extra v2 --extra benchmark python scripts/benchmark_v2.py --run --include-v1 --out /tmp/doc-v2-benchmark
```

`results.json` records field precision, recall, false-answer rate, evidence
errors, OCR/table misses, p50/p95 latency, and peak worker memory by parser
and model. Missing local weights or network access cause explicit combination
failures. Synthetic fixtures exercise the pipeline but cannot establish which
parser is more accurate on real business PDFs. The default parser/model should
only change after reviewed real-PDF results show better supported-value
accuracy while the warm 2–5 page p95 remains under two minutes.
