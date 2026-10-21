"""Versioned API that leaves the v1 request and response contracts intact."""

import asyncio
import logging
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.nuextract import TINY_MODEL
from app.v2_schema import parse_template
from app.v2_worker import WorkerError, worker


router = APIRouter(prefix="/api/v2")
logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_STORED_JOBS = 8
ModelName = Literal["numind/NuExtract-1.5-tiny", "numind/NuExtract-1.5-smol", "numind/NuExtract3"]
jobs: dict[str, asyncio.Task] = {}
busy = False


async def run_worker(*args):
    return await asyncio.to_thread(worker.run, *args)


class V2Response(BaseModel):
    model: str
    parser: str = "docling"
    pages_processed: int
    complete: bool
    data: dict[str, Any]
    evidence: dict[str, dict[str, Any]]
    warnings: list[str]
    metrics: dict[str, Any]


async def _read_request(pdf: UploadFile, template: str, fields: str) -> tuple[bytes, dict]:
    try:
        parsed_template = parse_template(template, fields)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not (pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(422, "Upload a PDF file.")
    content = await pdf.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF exceeds the 20 MB limit.")
    return content, parsed_template


def _reserve() -> None:
    global busy
    if busy:
        raise HTTPException(429, "An extraction is already running. Try again after it finishes.")
    busy = True


def _release() -> None:
    global busy
    busy = False


@router.post("/extract", response_model=V2Response)
async def extract_v2(
    pdf: UploadFile = File(...), template: str = Form(""), fields: str = Form(""),
    model: ModelName = Form(TINY_MODEL), force_ocr: bool = Form(False),
) -> dict:
    content, parsed_template = await _read_request(pdf, template, fields)
    _reserve()
    try:
        return await run_worker(content, parsed_template, model, force_ocr)
    except WorkerError as exc:
        raise HTTPException(502, str(exc)) from exc
    finally:
        _release()


async def _run_job(content: bytes, template: dict, model: str, force_ocr: bool) -> dict:
    try:
        result = await run_worker(content, template, model, force_ocr)
        return {"status": "completed" if result["complete"] else "incomplete", "result": result}
    except WorkerError as exc:
        return {"status": "failed", "detail": str(exc)}
    except Exception as exc:
        logger.error("V2 extraction job failed: %s", type(exc).__name__)
        return {"status": "failed", "detail": "Extraction failed. Check the server logs."}
    finally:
        _release()


@router.post("/extract/jobs", status_code=202)
async def start_v2_job(
    pdf: UploadFile = File(...), template: str = Form(""), fields: str = Form(""),
    model: ModelName = Form(TINY_MODEL), force_ocr: bool = Form(False),
) -> dict:
    content, parsed_template = await _read_request(pdf, template, fields)
    _reserve()
    for job_id, job in list(jobs.items()):
        if len(jobs) < MAX_STORED_JOBS:
            break
        if job.done():
            del jobs[job_id]
    job_id = uuid4().hex
    jobs[job_id] = asyncio.create_task(_run_job(content, parsed_template, model, force_ocr))
    return {"job_id": job_id, "status": "running"}


@router.get("/extract/jobs/{job_id}")
async def get_v2_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Extraction job not found. It may have been lost after a server restart.")
    if job.cancelled():
        return {"status": "failed", "detail": "Extraction job stopped."}
    return job.result() if job.done() else {"status": "running"}
