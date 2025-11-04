import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(path=None):
    """
    Load and preprocess traffic dataset.
    Returns X_train, X_test, y_train, y_test, scaler
    """

    # Default path relative to project root
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base_dir, "data", "traffic.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    expected_columns = {"volume", "temperature", "hour", "dayofweek", "speed"}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    X = df[["volume", "temperature", "hour", "dayofweek"]]
    y = df["speed"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_data()
    print("Sample X_train:", X_train[:2])
