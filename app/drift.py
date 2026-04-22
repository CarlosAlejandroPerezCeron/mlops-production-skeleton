import numpy as np

def detect_drift(reference_mean, new_data):
    new_mean = np.mean(new_data)
    drift_score = abs(reference_mean - new_mean)
    return drift_score