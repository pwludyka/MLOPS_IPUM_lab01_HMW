import numpy as np
from inference import (
    load_sentence_tarnsformer,
    load_model,
    calculate_embeddings,
    calculate_sentiment,
)


def test_load_sentence_tarnsformer() -> None:
    transformer = load_sentence_tarnsformer()
    assert transformer is not None


def test_load_model() -> None:
    model = load_model()
    assert model is not None


def test_inference() -> None:
    sample_strings = {
        "I am dissapointed": "negative",
        "I am very happy": "positive",
        "I am not sure I know how to write valuable tests": "neutral",
    }

    transformer = load_sentence_tarnsformer()
    model = load_model()
    for key in sample_strings:
        embeddings = calculate_embeddings(key, transformer)
        embeddings_arr = np.array(embeddings).reshape(1, -1)
        sentiment = calculate_sentiment(embeddings_arr, model)
        assert sentiment == sample_strings[key]
