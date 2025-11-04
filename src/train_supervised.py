from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
from preprocess import load_data
import os
from numpy import sqrt

def train_model():
    X_train, X_test, y_train, y_test, scaler = load_data()

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    dump(model, os.path.join(models_dir, "model.pkl"))
    dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_model()
