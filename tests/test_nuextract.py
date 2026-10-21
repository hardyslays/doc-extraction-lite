import json

import pytest

from app.nuextract import (
    NUEXTRACT3_MODEL,
    TINY_MODEL,
    LocalModelError,
    NuExtractFieldExtractor,
    load_extractor,
)


@pytest.mark.parametrize("model_name", [TINY_MODEL, NUEXTRACT3_MODEL])
def test_extractor_uses_model_specific_template_and_grounded_values(monkeypatch, model_name):
    extractor = NuExtractFieldExtractor(object(), object(), model_name=model_name, chunk_chars=40, chunk_overlap=10)
    calls = []

    def fake_generate(text, template):
        calls.append((text, json.loads(template)))
        return json.dumps({"Invoice number": "INV-4421", "Due date": "invented"})

    monkeypatch.setattr(extractor, "_generate", fake_generate)
    source = "Invoice number: INV-4421\n" + "details " * 10

    spans = extractor.extract(source, ["Invoice number", "Due date"])

    assert len(spans) == 1
    assert spans[0].value == "INV-4421"
    assert spans[0].confidence is None
    assert source[spans[0].start:spans[0].end] == spans[0].value
    expected_type = "" if model_name == TINY_MODEL else "verbatim-string"
    assert calls[0][1] == {"Invoice number": expected_type, "Due date": expected_type}


def test_extractor_rejects_invalid_json(monkeypatch):
    extractor = NuExtractFieldExtractor(object(), object(), model_name=TINY_MODEL)
    monkeypatch.setattr(extractor, "_generate", lambda text, template: "not-json")

    with pytest.raises(LocalModelError, match="invalid JSON"):
        extractor.extract("Invoice number: INV-4421", ["Invoice number"])


def test_extractor_accepts_fenced_json(monkeypatch):
    extractor = NuExtractFieldExtractor(object(), object(), model_name=NUEXTRACT3_MODEL)
    monkeypatch.setattr(
        extractor, "_generate", lambda text, template: '```json\n{"Invoice number": "INV-4421"}\n```'
    )

    spans = extractor.extract("Invoice number: INV-4421", ["Invoice number"])

    assert len(spans) == 1
    assert spans[0].value == "INV-4421"


def test_model_runtime_error_becomes_local_model_error():
    class BrokenProcessor:
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("out of memory")

    extractor = NuExtractFieldExtractor(object(), BrokenProcessor(), model_name=NUEXTRACT3_MODEL)

    with pytest.raises(LocalModelError, match="inference failed: out of memory"):
        extractor.extract("Invoice number: INV-4421", ["Invoice number"])


def test_model_loader_uses_separate_model_classes_and_reuses_cached_choice(monkeypatch):
    import transformers

    calls = []

    class FakeModel:
        def eval(self):
            return self

    def tiny_loader(model_name, **kwargs):
        calls.append(("tiny", model_name, kwargs))
        return FakeModel()

    def large_loader(model_name, **kwargs):
        calls.append(("large", model_name, kwargs))
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda model_name: object())
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda model_name, **kwargs: object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", tiny_loader)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", large_loader)
    load_extractor.cache_clear()
    try:
        assert load_extractor(TINY_MODEL) is load_extractor(TINY_MODEL)
        assert load_extractor(NUEXTRACT3_MODEL).name == NUEXTRACT3_MODEL
        assert [call[0] for call in calls] == ["tiny", "large"]
        assert calls[1][2]["device_map"] == "auto"
    finally:
        load_extractor.cache_clear()
