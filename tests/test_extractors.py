from app.extractors import GLiNERFieldExtractor


class FakeGLiNERModel:
    def __init__(self):
        self.calls = []

    def inference(self, texts, labels, **options):
        self.calls.append((texts, labels, options))
        predictions = []
        for text in texts:
            entities = []
            if "INV-4421" in text:
                start = text.index("INV-4421")
                entities.append(
                    {
                        "label": "invoice number",
                        "text": "ignored model text",
                        "score": 0.91,
                        "start": start,
                        "end": start + len("INV-4421"),
                    }
                )
            predictions.append(entities)
        return predictions


def test_gliner_extractor_returns_grounded_source_span():
    model = FakeGLiNERModel()
    extractor = GLiNERFieldExtractor(model, model_name="test-model")
    text = "Invoice number: INV-4421\nTotal: $99.00"

    matches = extractor.extract(text, ["Invoice number", "Total amount"])

    assert len(matches) == 1
    assert matches[0].field == "Invoice number"
    assert matches[0].value == "INV-4421"
    assert matches[0].confidence == 0.91
    assert text[matches[0].start:matches[0].end] == matches[0].value
    _, labels, options = model.calls[0]
    assert labels == ["Invoice number", "Total amount"]
    assert options["threshold"] == 0.5


def test_gliner_extractor_chunks_long_text_and_deduplicates_overlap():
    model = FakeGLiNERModel()
    extractor = GLiNERFieldExtractor(
        model,
        model_name="test-model",
        chunk_chars=40,
        chunk_overlap=20,
    )
    text = "Header information " + ("x" * 20) + " INV-4421 trailing details"

    matches = extractor.extract(text, ["Invoice number"])

    assert len(model.calls[0][0]) > 1
    assert len(matches) == 1
    assert matches[0].start == text.index("INV-4421")
