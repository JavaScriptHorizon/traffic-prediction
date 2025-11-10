from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from preprocess import load_data
import joblib, os

def train_mlp():
    X_train, X_test, y_train, y_test, scaler = load_data()

    model = MLPRegressor(hidden_layer_sizes=(64,32),
                         activation='relu',
                         solver='adam',
                         max_iter=500,
                         random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MLP Regressor Test MAE: {mae:.2f}")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "mlp_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print("MLP model and scaler saved!")

if __name__ == "__main__":
    train_mlp()
