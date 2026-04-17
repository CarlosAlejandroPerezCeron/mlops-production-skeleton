from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction():
    payload = {"feature1": 0.5, "feature2": 0.3, "feature3": 0.1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
