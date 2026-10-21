"""One reusable child process gives v2 jobs a hard crash and timeout boundary."""

import atexit
import multiprocessing as mp
from multiprocessing.connection import Connection
from threading import Lock

from app.nuextract import NUEXTRACT3_MODEL


class WorkerError(RuntimeError):
    pass


def _worker_main(connection: Connection) -> None:
    from app.v2_pipeline import run_v2_pipeline

    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        if request is None:
            break
        try:
            result = run_v2_pipeline(**request)
            connection.send({"ok": True, "result": result})
        except Exception as exc:
            # Library exceptions can contain input text; return only the failure stage.
            from app.v2_document import ParseError
            from app.v2_extract import ExtractionError

            if isinstance(exc, ParseError):
                detail = ("Structured OCR failed." if str(exc).startswith(
                    ("Structured OCR", "Could not read table", "PP-Structure"))
                    else "PDF parsing failed.")
            elif isinstance(exc, ExtractionError):
                detail = "The extraction model returned invalid output."
            else:
                detail = "The extraction worker failed. Check model dependencies and local weights."
            connection.send({"ok": False, "detail": detail})
    connection.close()


class V2Worker:
    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._lock = Lock()
        self._process = None
        self._connection = None

    def _start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        parent, child = self._context.Pipe()
        process = self._context.Process(target=_worker_main, args=(child,), daemon=True)
        process.start()
        child.close()
        self._process = process
        self._connection = parent

    def close(self) -> None:
        if self._process is not None:
            if self._process.is_alive():
                try:
                    self._connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=2)
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def run(self, pdf_bytes: bytes, template: dict, model: str, force_ocr: bool,
            parser: str = "docling") -> dict:
        timeout = 600 if model == NUEXTRACT3_MODEL else 180
        with self._lock:
            self._start()
            try:
                self._connection.send({"pdf_bytes": pdf_bytes, "template": template,
                                       "model": model, "force_ocr": force_ocr, "parser": parser})
                if not self._connection.poll(timeout):
                    self.close()
                    raise WorkerError(f"Extraction exceeded the {timeout}-second job limit.")
                message = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                self.close()
                raise WorkerError("Extraction worker stopped unexpectedly.") from exc
            if not message.get("ok"):
                self.close()
                raise WorkerError(message.get("detail", "Extraction failed."))
            return message["result"]


worker = V2Worker()
atexit.register(worker.close)
