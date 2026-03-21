from sentence_transformers import SentenceTransformer
import joblib
from typing import Any
import numpy as np


Sentiments = {0: "negative", 1: "neutral", 2: "positive"}


def load_sentence_tarnsformer() -> SentenceTransformer:
    transformer = SentenceTransformer("model/sentence_transformer.model")
    return transformer


def load_model() -> Any:
    with open("model/classifier.joblib", "rb") as file:
        model = joblib.load(file)
    return model


def calculate_embeddings(
    sentence: dict[str, Any], transformer: SentenceTransformer
) -> np.ndarray:
    embeddings = transformer.encode(str(sentence))
    return embeddings


def calculate_sentiment(embedding: np.ndarray, model: Any) -> str:
    sentiment = model.predict(embedding)
    return Sentiments[sentiment[0]]
