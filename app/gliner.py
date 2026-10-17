import os
from functools import lru_cache


GLINER_MODEL = os.getenv("GLINER_MODEL", "gliner-community/gliner_large-v2.5")
GLINER_MODEL_FILES = (
    "added_tokens.json",
    "gliner_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


class LocalModelError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model():
    """Load the local GLiNER checkpoint once per application process."""
    try:
        from gliner import GLiNER
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise LocalModelError("GLiNER is not installed. Run `uv sync` before starting the app.") from exc

    try:
        checkpoint = snapshot_download(GLINER_MODEL, allow_patterns=GLINER_MODEL_FILES)
        return GLiNER.from_pretrained(checkpoint)
    except OSError as exc:
        raise LocalModelError(
            f"Could not load GLiNER model '{GLINER_MODEL}'. Check network access or the local Hugging Face cache."
        ) from exc
