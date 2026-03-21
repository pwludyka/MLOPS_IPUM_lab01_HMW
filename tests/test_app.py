from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_predict() -> None:
    response = client.post(
        "/predict", json={"text": "What a great MLOps lecture, I am very satisfied"}
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}


def test_input_empty_string() -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    assert "detail" in response.json()
