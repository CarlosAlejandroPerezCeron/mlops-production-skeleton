from fastapi import FastAPI
from prometheus_client import generate_latest
from fastapi.responses import Response
from app.model import load_model
from app.schema import PredictionRequest
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY
import time
import numpy as np

app = FastAPI()
model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    start = time.time()
    REQUEST_COUNT.inc()

    features = np.array([[request.feature1, request.feature2, request.feature3]])
    prediction = model.predict(features)[0]

    REQUEST_LATENCY.observe(time.time() - start)

    return {"prediction": int(prediction)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")