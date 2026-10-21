"""Token-bounded NuExtract calls and evidence-first result validation."""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.nuextract import load_extractor
from app.v2_document import Document
from app.v2_schema import blank_data, model_template, pointer


MAX_CHUNK_TOKENS = 1800
MAX_CHUNKS = 12
MAX_PRIMARY_CHUNKS = 8
MAX_FOLLOWUP_CALLS = 4
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b",
    re.IGNORECASE,
)
DATE_TIME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?\b")


class ExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    page: int
    text: str


def _token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except (AttributeError, TypeError):
        try:
            return len(tokenizer.tokenizer.encode(text, add_special_tokens=False))
        except (AttributeError, TypeError):
            return max(1, len(text) // 3)


def make_chunks(document: Document, tokenizer: Any, *, limit: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        current: list[str] = []
        header = ""
        for block in page.blocks:
            if block.kind != "table_row":
                header = ""
            elif not header:
                header = block.text
            unit = block.text
            if block.kind == "table_row" and header and unit != header:
                unit = header + "\n" + unit
            if not unit.strip():
                continue
            if current and _token_count(tokenizer, "\n".join(current + [unit])) > limit:
                chunks.append(Chunk(page.number, "\n".join(current)))
                current = []
            if _token_count(tokenizer, unit) <= limit:
                current.append(unit)
                continue
            words = unit.split()
            part: list[str] = []
            for word in words:
                if _token_count(tokenizer, word) > limit:
                    if part:
                        chunks.append(Chunk(page.number, " ".join(part)))
                        part = []
                    remainder = word
                    while remainder:
                        length = min(len(remainder), limit)
                        while length > 1 and _token_count(tokenizer, remainder[:length]) > limit:
                            length //= 2
                        chunks.append(Chunk(page.number, remainder[:length]))
                        remainder = remainder[length:]
                    continue
                if part and _token_count(tokenizer, " ".join(part + [word])) > limit:
                    chunks.append(Chunk(page.number, " ".join(part)))
                    part = []
                part.append(word)
            if part:
                current.append(" ".join(part))
        if current:
            chunks.append(Chunk(page.number, "\n".join(current)))
    return chunks


def _date_value(text: str) -> str | None:
    value = text.strip().replace(".", "")
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    if "/" in value:
        parts = value.split("/")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            first, second = int(parts[0]), int(parts[1])
            if first <= 12 and second <= 12:
                return None
            fmt = "%d/%m/%Y" if first > 12 else "%m/%d/%Y"
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                pass
    return None


def _date_time_value(text: str) -> str | None:
    try:
        value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if "T" not in text and " " not in text:
        return None
    return value.isoformat()


def _number_value(value: Any, *, integer: bool) -> int | float | None:
    if isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    if integer:
        return int(number) if number == number.to_integral_value() else None
    return float(number)


def _ground(value: Any, field_type: str, text: str) -> tuple[Any, str] | None:
    if value is None or isinstance(value, bool):
        return None
    if field_type in {"string", "verbatim-string", "currency"}:
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()
        if field_type == "currency" and not re.fullmatch(r"[A-Z]{3}", value):
            return None
        return (value, value) if value in text else None
    if field_type in {"number", "integer"}:
        target = _number_value(value, integer=field_type == "integer")
        if target is None:
            return None
        for match in NUMBER_RE.finditer(text):
            source = _number_value(match.group(), integer=field_type == "integer")
            if source is not None and source == target:
                return target, match.group()
        return None
    if field_type == "date-time":
        if not isinstance(value, str) or not DATE_TIME_RE.fullmatch(value.strip()):
            return None
        target = _date_time_value(value)
        for match in DATE_TIME_RE.finditer(text):
            if target is not None and _date_time_value(match.group()) == target:
                return target, match.group()
        return None
    if field_type == "date":
        if not isinstance(value, str):
            return None
        target = _date_value(value)
        if target is None:
            return None
        for match in DATE_RE.finditer(text):
            if _date_value(match.group()) == target:
                return target, match.group()
        return None
    return None


def _validate(node: Any, template: Any, chunk: Chunk, path: tuple[str | int, ...], evidence: dict) -> Any:
    if isinstance(template, str):
        grounded = _ground(node, template, chunk.text)
        if grounded is None:
            return None
        result, quote = grounded
        evidence[pointer(path)] = {"quote": quote, "page": chunk.page}
        return result
    if isinstance(template, dict):
        source = node if isinstance(node, dict) else {}
        return {key: _validate(source.get(key), child, chunk, path + (key,), evidence)
                for key, child in template.items()}
    if not isinstance(node, list):
        return None
    items = []
    item_occurrences: dict[tuple[str, ...], int] = {}
    for value in node[:100]:
        index = len(items)
        item = _validate(value, template[0], chunk, path + (index,), evidence)
        item_prefix = pointer(path + (index,)) + "/"
        quotes = [source["quote"] for key, source in evidence.items() if key.startswith(item_prefix)]
        signature = tuple(sorted(quotes))
        matching_lines = sum(all(quote in line for quote in quotes)
                             for line in chunk.text.splitlines()) if quotes else 0
        if quotes and matching_lines <= item_occurrences.get(signature, 0):
            for key in list(evidence):
                if key.startswith(item_prefix):
                    del evidence[key]
            continue
        if quotes:
            item_occurrences[signature] = item_occurrences.get(signature, 0) + 1
        items.append(item)
    return items if _has_value(items) else None


def _has_value(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_value(item) for item in value)
    return value is not None


def _merge(template: Any, candidates: list[tuple[Any, dict, tuple[str | int, ...]]],
           path: tuple[str | int, ...], evidence: dict, conflicts: list[str]) -> Any:
    if isinstance(template, dict):
        return {key: _merge(child, [(value.get(key), ev, old + (key,)) for value, ev, old in candidates
                                    if isinstance(value, dict)], path + (key,), evidence, conflicts)
                for key, child in template.items()}
    if isinstance(template, list):
        items: list[Any] = []
        seen_in_earlier_chunks: set[tuple[str, tuple[int, ...]]] = set()
        for value, ev, old in candidates:
            if not isinstance(value, list):
                continue
            current_keys: set[tuple[str, tuple[int, ...]]] = set()
            for old_index, item in enumerate(value):
                if not _has_value(item):
                    continue
                old_prefix = pointer(old + (old_index,))
                pages = tuple(sorted({source["page"] for source_path, source in ev.items()
                                      if source_path.startswith(old_prefix + "/")}))
                key = (json.dumps(item, sort_keys=True, ensure_ascii=False), pages)
                if key in seen_in_earlier_chunks:
                    continue
                current_keys.add(key)
                new_index = len(items)
                items.append(item)
                new_prefix = pointer(path + (new_index,))
                for source_path, source in ev.items():
                    if source_path.startswith(old_prefix + "/") or source_path == old_prefix:
                        evidence[new_prefix + source_path[len(old_prefix):]] = source
            seen_in_earlier_chunks.update(current_keys)
        return items or None
    values = [(value, ev.get(pointer(old))) for value, ev, old in candidates if value is not None]
    distinct = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value, _ in values}
    if len(distinct) > 1:
        conflicts.append(pointer(path))
        return None
    if values:
        value, source = values[0]
        if source:
            evidence[pointer(path)] = source
        return value
    return None


def _field_terms(template: Any) -> set[str]:
    if isinstance(template, str):
        return set()
    if isinstance(template, list):
        return _field_terms(template[0])
    return {word.lower() for key, child in template.items()
            for word in key.replace("_", " ").split() if len(word) > 2} | {
                term for child in template.values() for term in _field_terms(child)}


def _unresolved(template: Any, data: Any, path: tuple[str | int, ...],
                conflicts: set[str]) -> Any:
    if pointer(path) in conflicts:
        return None
    if isinstance(template, str):
        return template if data is None else None
    if isinstance(template, list):
        if data is None or any(_unresolved(template[0], item, path, conflicts) is not None
                               for item in data):
            return template
        return None
    children = {key: _unresolved(child, (data or {}).get(key), path + (key,), conflicts)
                for key, child in template.items()}
    return {key: child for key, child in children.items() if child is not None} or None


def _run_chunk(extractor: Any, chunk: Chunk, template: dict, model_name: str) -> tuple[Any, dict, tuple]:
    prompt_template = json.dumps(model_template(template, model_name), ensure_ascii=False)
    try:
        output = extractor._generate(chunk.text, prompt_template).strip()
        if output.startswith("```"):
            output = output.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(output)
    except (json.JSONDecodeError, IndexError) as exc:
        raise ExtractionError(f"{model_name} returned invalid JSON on page {chunk.page}.") from exc
    if not isinstance(parsed, dict):
        raise ExtractionError(f"{model_name} returned an invalid object on page {chunk.page}.")
    evidence: dict[str, dict] = {}
    return _validate(parsed, template, chunk, (), evidence), evidence, ()


def extract_structured(document: Document, template: dict, model_name: str) -> tuple[dict, dict, int, list[str]]:
    extractor = load_extractor(model_name)
    tokenizer = extractor._processor
    full_prompt = json.dumps(model_template(template, model_name), ensure_ascii=False)
    chunk_limit = max(256, min(MAX_CHUNK_TOKENS, 2400 - _token_count(tokenizer, full_prompt) - 128))
    chunks = make_chunks(document, tokenizer, limit=chunk_limit)
    if not chunks:
        return blank_data(template), {}, 0, []
    terms = _field_terms(template)
    chunks.sort(key=lambda chunk: -sum(term in chunk.text.lower() for term in terms))
    warnings = []
    candidates: list[tuple[Any, dict, tuple[str | int, ...]]] = []
    primary = chunks[:MAX_PRIMARY_CHUNKS]
    for chunk in primary:
        candidates.append(_run_chunk(extractor, chunk, template, model_name))
    combined_evidence: dict[str, dict] = {}
    conflicts: list[str] = []
    data = _merge(template, candidates, (), combined_evidence, conflicts)
    unresolved = _unresolved(template, data, (), set(conflicts))
    followups = 0
    if unresolved is not None and not conflicts and len(chunks) > len(primary):
        followup_terms = _field_terms(unresolved)
        ranked = sorted(chunks[MAX_PRIMARY_CHUNKS:MAX_CHUNKS],
                        key=lambda chunk: -sum(term in chunk.text.lower() for term in followup_terms))
        for chunk in ranked[:MAX_FOLLOWUP_CALLS]:
            candidates.append(_run_chunk(extractor, chunk, unresolved, model_name))
            followups += 1
        combined_evidence = {}
        conflicts = []
        data = _merge(template, candidates, (), combined_evidence, conflicts)
    if len(chunks) > len(primary):
        warnings.append(f"Only {len(primary)} of {len(chunks)} document chunks were fully examined.")
    if conflicts:
        warnings.append("Conflicting source values were discarded: " + ", ".join(conflicts))
    return data, combined_evidence, len(primary) + followups, warnings
