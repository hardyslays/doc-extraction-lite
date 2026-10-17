# doc-extraction-lite
Configurable document-extraction service built around non-generative AI models.
## Document Field Extractor

A local pipeline for extracting named fields from text-native PDFs. It uses GLiNER to identify source spans rather than generate values, so every returned value can be traced directly to the document text.

It uses [`gliner-community/gliner_large-v2.5`](https://huggingface.co/gliner-community/gliner_large-v2.5) by default. This 1.85 GB encoder-only checkpoint runs locally on CPU, fits comfortably on a 16 GB machine, and is downloaded once to the local Hugging Face cache. The loader fetches only the FP32 files used for CPU inference, not the optional FP16/BF16 variants. No document content is sent to a hosted API.

### Run it

1. Install the locked application and CPU-only PyTorch runtime with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync --extra dev
   ```

2. Start the web app:

   ```bash
   uv run uvicorn app.main:app --reload
   ```

   Visit `http://127.0.0.1:8000`.

On its first start, the app downloads and validates the GLiNER checkpoint. Optional configuration is in `.env.example`; export `GLINER_MODEL` before starting the server.

### Current pipeline

```text
PDF upload → PyPDF text per page → GLiNER source-span extraction
          → evidence validation → highest-confidence result per requested field → UI
```

The extractor sits behind a small `FieldExtractor` protocol, so another model or architecture can replace GLiNER without changing PDF parsing, candidate validation, result selection, or the API schema. Missing fields are returned with a `null` value and confidence `0`.

### API

Send `POST /api/extract` as multipart form data with a PDF in `pdf` and newline-separated field names in `fields`. The response is:

```json
{
  "model": "gliner-community/gliner_large-v2.5",
  "pages_processed": 1,
  "results": [
    {
      "field": "Invoice number",
      "value": "INV-4421",
      "confidence": 0.92,
      "evidence": "INV-4421",
      "page": 1
    }
  ]
}
```

### Verify

```bash
uv run pytest
```
