from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExtractedSpan:
    field: str
    value: str
    confidence: float | None
    start: int
    end: int


class FieldExtractor(Protocol):
    name: str

    def extract(self, text: str, fields: list[str]) -> list[ExtractedSpan]: ...
