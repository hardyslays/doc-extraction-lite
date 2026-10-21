import asyncio
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.extractors import FieldExtractor
from app.models import ExtractionResponse, ExtractionResult, FieldCandidate, PipelineMetrics, grounded_candidate, select_best
from app.nuextract import MODEL_OPTIONS, TINY_MODEL, LocalModelError, load_extractor
from app.ocr import LocalOCRError
from app.pdf import extract_pages
from app.v2_api import router as v2_router


MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_STORED_JOBS = 8
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
model_load_lock = Lock()
jobs: dict[str, asyncio.Task[dict]] = {}
logger = logging.getLogger(__name__)
ExtractionModel = Literal["numind/NuExtract-1.5-tiny", "numind/NuExtract3"]


def get_extractor(model_name: str) -> FieldExtractor:
    with model_load_lock:
        return load_extractor(model_name)


app = FastAPI(title="Document Field Extractor")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(v2_router)


def parse_fields(raw_fields: str) -> list[str]:
    fields = [line.strip() for line in raw_fields.splitlines() if line.strip()]
    # Keep ordering while avoiding duplicate model requests.
    return list(dict.fromkeys(fields))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    from app.nuextract import V2_MODEL_OPTIONS

    return templates.TemplateResponse(request, "index.html", {"models": MODEL_OPTIONS, "v2_models": V2_MODEL_OPTIONS})


@app.post("/api/extract", response_model=ExtractionResponse)
async def extract(
    pdf: UploadFile = File(...),
    fields: str = Form(""),
    model: ExtractionModel = Form(TINY_MODEL),
    force_ocr: bool = Form(False),
) -> ExtractionResponse:
    started = perf_counter()
    content, requested_fields = await read_upload(pdf, fields)
    return await run_in_threadpool(run_pipeline, content, requested_fields, model, force_ocr, started)


@app.post("/api/extract/jobs", status_code=202)
async def start_extraction_job(
    pdf: UploadFile = File(...),
    fields: str = Form(""),
    model: ExtractionModel = Form(TINY_MODEL),
    force_ocr: bool = Form(False),
) -> dict:
    started = perf_counter()
    content, requested_fields = await read_upload(pdf, fields)
    if any(not job.done() for job in jobs.values()):
        raise HTTPException(429, "An extraction is already running. Try again after it finishes.")
    for job_id, job in list(jobs.items()):
        if len(jobs) < MAX_STORED_JOBS:
            break
        if job.done():
            del jobs[job_id]
    job_id = uuid4().hex
    jobs[job_id] = asyncio.create_task(run_job(content, requested_fields, model, force_ocr, started))
    return {"job_id": job_id, "status": "running"}


@app.get("/api/extract/jobs/{job_id}")
async def get_extraction_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Extraction job not found. It may have been lost after a server restart.")
    if job.cancelled():
        return {"status": "failed", "detail": "Extraction job stopped."}
    if not job.done():
        return {"status": "running"}
    return job.result()


async def read_upload(pdf: UploadFile, fields: str) -> tuple[bytes, list[str]]:
    requested_fields = parse_fields(fields)
    if not requested_fields:
        raise HTTPException(422, "Add at least one field, one per line.")
    if len(requested_fields) > 20:
        raise HTTPException(422, "Use at most 20 fields per request.")
    if not (pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(422, "Upload a PDF file.")
    content = await pdf.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF exceeds the 20 MB limit.")
    return content, requested_fields


async def run_job(
    content: bytes, requested_fields: list[str], model: str, force_ocr: bool, started: float
) -> dict:
    try:
        result = await run_in_threadpool(run_pipeline, content, requested_fields, model, force_ocr, started)
        return {"status": "completed", "result": result.model_dump()}
    except HTTPException as exc:
        return {"status": "failed", "detail": str(exc.detail)}
    except Exception:
        logger.exception("Extraction job failed")
        return {"status": "failed", "detail": "Extraction failed. Check the server logs."}


def run_pipeline(
    content: bytes, requested_fields: list[str], model: str, force_ocr: bool, started: float
) -> ExtractionResponse:
    try:
        pages = extract_pages(content, force_ocr=force_ocr)
    except LocalOCRError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Unable to read PDF: {exc}") from exc
    text_extraction_seconds = perf_counter() - started

    model_started = perf_counter()
    try:
        extractor = get_extractor(model)
    except LocalModelError as exc:
        raise HTTPException(503, str(exc)) from exc
    model_load_seconds = perf_counter() - model_started

    candidates: dict[str, list[FieldCandidate]] = {field: [] for field in requested_fields}
    page_starts: list[int] = []
    offset = 0
    for page_text in pages:
        page_starts.append(offset)
        offset += len(page_text) + 2
    document_text = "\n\n".join(pages)
    inference_started = perf_counter()
    try:
        spans = extractor.extract(document_text, requested_fields)
    except LocalModelError as exc:
        raise HTTPException(502, str(exc)) from exc
    inference_seconds = perf_counter() - inference_started
    for span in spans:
        for page_number, (page_start, page_text) in enumerate(zip(page_starts, pages), start=1):
            if not page_start <= span.start < span.end <= page_start + len(page_text):
                continue
            candidate = grounded_candidate(
                field=span.field,
                value=span.value,
                confidence=span.confidence,
                evidence=span.value,
                page=page_number,
                page_text=page_text,
            )
            if candidate is not None and candidate.field in candidates:
                candidates[candidate.field].append(candidate)
            break

    results = []
    for field in requested_fields:
        best = select_best(candidates[field])
        results.append(
            ExtractionResult(
                field=field,
                value=best.value if best else None,
                confidence=best.confidence if best else None,
                evidence=best.evidence if best else "",
                page=best.page if best else None,
            )
        )

    return ExtractionResponse(
        model=extractor.name,
        pages_processed=len(pages),
        metrics=PipelineMetrics(
            total_seconds=perf_counter() - started,
            text_extraction_seconds=text_extraction_seconds,
            model_load_seconds=model_load_seconds,
            inference_seconds=inference_seconds,
            fields_requested=len(requested_fields),
            fields_extracted=sum(result.value is not None for result in results),
        ),
        results=results,
    )
