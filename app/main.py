from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest
import numpy as np
import time

from app.model import load_model
from app.schema import PredictionRequest
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY
from app.feature_store import get_features
from app.drift import detect_drift
from app.logger import setup_logger
from app.versioning import get_active_model_version

app = FastAPI()
logger = setup_logger()
model = load_model()
reference_mean = 0.5

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    start = time.time()
    REQUEST_COUNT.inc()

    features = np.array([get_features(request)])
    prediction = model.predict(features)[0]

    drift_score = detect_drift(reference_mean, features.flatten())

    logger.info(f"Prediction made | Drift score: {drift_score}")

    REQUEST_LATENCY.observe(time.time() - start)

    return {
        "prediction": int(prediction),
        "model_version": get_active_model_version(),
        "drift_score": drift_score
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
