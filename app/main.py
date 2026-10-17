from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.extractors import FieldExtractor, GLiNERFieldExtractor
from app.gliner import GLINER_MODEL, LocalModelError, load_model
from app.models import ExtractionResponse, ExtractionResult, FieldCandidate, grounded_candidate, select_best
from app.pdf import extract_pages


MAX_PDF_BYTES = 20 * 1024 * 1024
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@lru_cache(maxsize=1)
def get_extractor() -> FieldExtractor:
    return GLiNERFieldExtractor(load_model(), model_name=GLINER_MODEL)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_extractor()
    except LocalModelError as exc:
        raise RuntimeError(str(exc)) from exc
    yield


app = FastAPI(title="Document Field Extractor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def parse_fields(raw_fields: str) -> list[str]:
    fields = [line.strip() for line in raw_fields.splitlines() if line.strip()]
    # Keep ordering while avoiding duplicate model requests.
    return list(dict.fromkeys(fields))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"model": GLINER_MODEL})


@app.post("/api/extract", response_model=ExtractionResponse)
async def extract(
    pdf: UploadFile = File(...),
    fields: str = Form(""),
    extractor: FieldExtractor = Depends(get_extractor),
) -> ExtractionResponse:
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
    try:
        pages = extract_pages(content)
    except Exception as exc:
        raise HTTPException(422, f"Unable to read PDF: {exc}") from exc

    candidates: dict[str, list[FieldCandidate]] = {field: [] for field in requested_fields}
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        spans = await run_in_threadpool(extractor.extract, page_text, requested_fields)
        for span in spans:
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

    results = []
    for field in requested_fields:
        best = select_best(candidates[field])
        results.append(
            ExtractionResult(
                field=field,
                value=best.value if best else None,
                confidence=best.confidence if best else 0,
                evidence=best.evidence if best else "",
                page=best.page if best else None,
            )
        )

    return ExtractionResponse(model=extractor.name, pages_processed=len(pages), results=results)
