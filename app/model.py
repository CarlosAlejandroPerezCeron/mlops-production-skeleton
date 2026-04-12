import joblib
import os
from sklearn.linear_model import LogisticRegression
import numpy as np

MODEL_PATH = "model.joblib"

def train_and_save():
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)
