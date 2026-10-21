import json
from functools import lru_cache
from threading import Lock

from app.extractors import ExtractedSpan


TINY_MODEL = "numind/NuExtract-1.5-tiny"
SMOL_MODEL = "numind/NuExtract-1.5-smol"
NUEXTRACT3_MODEL = "numind/NuExtract3"
MODEL_OPTIONS = (TINY_MODEL, NUEXTRACT3_MODEL)
V2_MODEL_OPTIONS = (TINY_MODEL, SMOL_MODEL, NUEXTRACT3_MODEL)


class LocalModelError(RuntimeError):
    pass


class NuExtractFieldExtractor:
    """Run one of NuMind's template-guided models and keep only quoted spans."""

    def __init__(self, model, processor, *, model_name: str, chunk_chars: int = 6_000, chunk_overlap: int = 400):
        if chunk_chars <= 0 or not 0 <= chunk_overlap < chunk_chars:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        self._model = model
        self._processor = processor
        self._lock = Lock()
        self._chunk_chars = chunk_chars
        self._chunk_overlap = chunk_overlap
        self.name = model_name

    def extract(self, text: str, fields: list[str]) -> list[ExtractedSpan]:
        if not text or not fields:
            return []

        template = json.dumps(
            {field: "" if self.name == TINY_MODEL else "verbatim-string" for field in fields},
            ensure_ascii=False,
        )
        matches: dict[tuple[str, int, int], ExtractedSpan] = {}
        for chunk_start in range(0, len(text), self._chunk_chars - self._chunk_overlap):
            chunk = text[chunk_start:chunk_start + self._chunk_chars]
            output = self._generate(chunk, template)
            if output.startswith("```"):
                output = output.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                values = json.loads(output)
            except json.JSONDecodeError as exc:
                raise LocalModelError(f"{self.name} returned invalid JSON.") from exc
            if not isinstance(values, dict):
                raise LocalModelError(f"{self.name} returned JSON that is not an object.")

            for field in fields:
                value = values.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip()
                local_start = chunk.find(value)
                if local_start < 0:
                    continue
                start = chunk_start + local_start
                end = start + len(value)
                matches[(field, start, end)] = ExtractedSpan(field, value, None, start, end)

            if chunk_start + self._chunk_chars >= len(text):
                break
        return list(matches.values())

    def _generate(self, text: str, template: str) -> str:
        import torch

        try:
            with self._lock, torch.inference_mode():
                if self.name in {TINY_MODEL, SMOL_MODEL}:
                    prompt = f"<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"
                    inputs = self._processor(prompt, return_tensors="pt").to(self._model.device)
                else:
                    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                    inputs = self._processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        template=template,
                        enable_thinking=False,
                    ).to(self._model.device)
                generated = self._model.generate(**inputs, max_new_tokens=2048, do_sample=False)
                output_ids = generated[:, inputs["input_ids"].shape[1]:]
                return self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        except (MemoryError, RuntimeError, TypeError, ValueError) as exc:
            raise LocalModelError(f"{self.name} inference failed: {exc}") from exc


@lru_cache(maxsize=1)
def load_extractor(model_name: str) -> NuExtractFieldExtractor:
    """Load only the selected checkpoint; evict the previous one when switching."""
    if model_name not in V2_MODEL_OPTIONS:
        raise ValueError("Unknown extraction model.")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        if model_name in {TINY_MODEL, SMOL_MODEL}:
            processor = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).eval()
        else:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            ).eval()
        return NuExtractFieldExtractor(model, processor, model_name=model_name)
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        raise LocalModelError(
            f"Could not load {model_name}. Check the model download, dependencies, and available memory: {exc}"
        ) from exc
