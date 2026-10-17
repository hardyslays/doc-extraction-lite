from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExtractedSpan:
    field: str
    value: str
    confidence: float
    start: int
    end: int


class FieldExtractor(Protocol):
    name: str

    def extract(self, text: str, fields: list[str]) -> list[ExtractedSpan]: ...


@dataclass(frozen=True)
class _TextChunk:
    text: str
    start: int


class GLiNERFieldExtractor:
    """Adapt GLiNER's entity predictions to source-grounded field spans."""

    def __init__(
        self,
        model,
        *,
        model_name: str,
        threshold: float = 0.5,
        chunk_chars: int = 2_000,
        chunk_overlap: int = 200,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if chunk_chars <= 0 or not 0 <= chunk_overlap < chunk_chars:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        self._model = model
        self._threshold = threshold
        self._chunk_chars = chunk_chars
        self._chunk_overlap = chunk_overlap
        self.name = model_name

    def extract(self, text: str, fields: list[str]) -> list[ExtractedSpan]:
        if not text or not fields:
            return []

        chunks = list(self._chunks(text))
        predictions = self._model.inference(
            [chunk.text for chunk in chunks],
            fields,
            flat_ner=True,
            threshold=self._threshold,
            multi_label=False,
        )
        field_names = {field.casefold(): field for field in fields}
        matches: dict[tuple[str, int, int], ExtractedSpan] = {}

        for chunk, entities in zip(chunks, predictions, strict=True):
            for entity in entities:
                match = self._to_span(entity, chunk, text, field_names)
                if match is None:
                    continue
                key = (match.field, match.start, match.end)
                previous = matches.get(key)
                if previous is None or match.confidence > previous.confidence:
                    matches[key] = match

        return list(matches.values())

    def _to_span(
        self,
        entity: object,
        chunk: _TextChunk,
        source_text: str,
        field_names: dict[str, str],
    ) -> ExtractedSpan | None:
        if not isinstance(entity, dict):
            return None
        label = entity.get("label")
        local_start = entity.get("start")
        local_end = entity.get("end")
        if not isinstance(label, str) or not isinstance(local_start, int) or not isinstance(local_end, int):
            return None
        field = field_names.get(label.casefold())
        if field is None or not 0 <= local_start < local_end <= len(chunk.text):
            return None

        start = chunk.start + local_start
        end = chunk.start + local_end
        value = source_text[start:end]
        if not value.strip():
            return None
        try:
            confidence = float(entity.get("score"))
        except (TypeError, ValueError):
            return None

        return ExtractedSpan(
            field=field,
            value=value,
            confidence=max(0.0, min(1.0, confidence)),
            start=start,
            end=end,
        )

    def _chunks(self, text: str):
        start = 0
        while start < len(text):
            end = min(start + self._chunk_chars, len(text))
            if end < len(text):
                search_from = start + self._chunk_chars // 2
                boundary = max(text.rfind("\n", search_from, end), text.rfind(" ", search_from, end))
                if boundary > start:
                    end = boundary
            yield _TextChunk(text=text[start:end], start=start)
            if end == len(text):
                break
            start = max(start + 1, end - self._chunk_overlap)
