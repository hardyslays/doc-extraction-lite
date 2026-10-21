"""The deliberately small template language exposed by the v2 API."""

import json
from typing import Any


LEAF_TYPES = {"verbatim-string", "string", "date", "date-time", "number", "integer", "currency"}
MAX_LEAVES = 20


def parse_template(raw_template: str, raw_fields: str) -> dict[str, Any]:
    if bool(raw_template.strip()) == bool(raw_fields.strip()):
        raise ValueError("Provide exactly one of template or fields.")
    if raw_fields.strip():
        names = list(dict.fromkeys(line.strip() for line in raw_fields.splitlines() if line.strip()))
        if not 1 <= len(names) <= MAX_LEAVES:
            raise ValueError("Use between 1 and 20 fields.")
        return {name: "verbatim-string" for name in names}
    try:
        template = json.loads(raw_template)
    except json.JSONDecodeError as exc:
        raise ValueError("Template must be valid JSON.") from exc
    if not isinstance(template, dict) or not template:
        raise ValueError("Template must be a nonempty JSON object.")
    count = _validate(template, 0)
    if not 1 <= count <= MAX_LEAVES:
        raise ValueError("Template must contain between 1 and 20 leaf fields.")
    return template


def _validate(node: Any, depth: int) -> int:
    if depth > 4:
        raise ValueError("Template nesting is limited to four levels.")
    if isinstance(node, str):
        if node not in LEAF_TYPES:
            raise ValueError(f"Unsupported template type: {node}")
        return 1
    if isinstance(node, dict) and node:
        if any(not isinstance(key, str) or not key.strip() for key in node):
            raise ValueError("Template keys must be nonempty strings.")
        return sum(_validate(value, depth + 1) for value in node.values())
    if isinstance(node, list) and len(node) == 1:
        return _validate(node[0], depth + 1)
    raise ValueError("Template values must be a field type, object, or one-item array.")


def blank_data(template: Any) -> Any:
    if isinstance(template, str):
        return None
    if isinstance(template, list):
        return None
    return {key: blank_data(value) for key, value in template.items()}


def model_template(template: Any, model_name: str) -> Any:
    """Older NuExtract checkpoints use empty leaves; v3 supports typed leaves."""
    if isinstance(template, str):
        return template if model_name == "numind/NuExtract3" else ""
    if isinstance(template, list):
        return [model_template(template[0], model_name)]
    return {key: model_template(value, model_name) for key, value in template.items()}


def pointer(path: tuple[str | int, ...]) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)
