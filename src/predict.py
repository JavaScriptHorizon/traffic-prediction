from joblib import load
import numpy as np
import os

def predict_speed_rf(volume, temperature, hour, dayofweek):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    model = load(model_path)
    scaler = load(scaler_path)
    X = np.array([[volume, temperature, hour, dayofweek]])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])

def predict_speed_mlp(volume, temperature, hour, dayofweek):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "mlp_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    model = load(model_path)
    scaler = load(scaler_path)
    X = np.array([[volume, temperature, hour, dayofweek]])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])

