from app import gliner


def test_default_gliner_model_is_large_cpu_capable_checkpoint():
    assert gliner.GLINER_MODEL == "gliner-community/gliner_large-v2.5"


def test_cpu_loader_excludes_optional_half_precision_checkpoints():
    assert "pytorch_model.bin" in gliner.GLINER_MODEL_FILES
    assert "model.fp16.safetensors" not in gliner.GLINER_MODEL_FILES
    assert "model.bf16.safetensors" not in gliner.GLINER_MODEL_FILES


def test_model_loader_downloads_only_cpu_files_once(monkeypatch):
    from gliner import GLiNER
    import huggingface_hub

    loaded_model = object()
    calls = []

    def fake_snapshot_download(model, *, allow_patterns):
        calls.append((model, allow_patterns))
        return "/tmp/gliner-checkpoint"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(GLiNER, "from_pretrained", lambda checkpoint: loaded_model)
    gliner.load_model.cache_clear()

    assert gliner.load_model() is loaded_model
    assert gliner.load_model() is loaded_model
    assert calls == [(gliner.GLINER_MODEL, gliner.GLINER_MODEL_FILES)]

    gliner.load_model.cache_clear()
