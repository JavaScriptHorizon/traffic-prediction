from joblib import load
import numpy as np
import os

def predict_speed(volume, temperature, hour, dayofweek):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = load(model_path)
    scaler = load(scaler_path)

    X = np.array([[volume, temperature, hour, dayofweek]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    return float(prediction[0])


if __name__ == "__main__":
    result = predict_speed(150, 28.5, 17, 4)
    print(f"Predicted traffic speed: {result:.2f} km/h")
