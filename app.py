from fastapi import FastAPI
from api.models.sentiment import PredictRequest, PredictResponse
from inference import (
    load_sentence_tarnsformer,
    load_model,
    calculate_embeddings,
    calculate_sentiment,
)
import numpy as np

model = load_model()
sentence_transformer = load_sentence_tarnsformer()

app = FastAPI()


@app.post("/predict")
def Predict(sentence: PredictRequest) -> PredictResponse:
    embeddings = calculate_embeddings(sentence.text, sentence_transformer)
    embeddings_arr = np.array(embeddings).reshape(1, -1)
    sentiment = calculate_sentiment(embeddings_arr, model)
    return PredictResponse(prediction=sentiment)
