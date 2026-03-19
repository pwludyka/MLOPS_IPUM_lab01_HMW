from fastapi import FastAPI
from api.models.sentiment import PredictRequest, PredictResponse

app = FastAPI()


@app.post("/predict")
def Predict(text: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction="Positive")
