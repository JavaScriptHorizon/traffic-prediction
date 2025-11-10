import streamlit as st
from src.predict import predict_speed_rf, predict_speed_mlp
from PIL import Image
import os

st.set_page_config(page_title="Traffic Speed Predictor", page_icon="🚦")
st.title("🚗 Traffic Speed Prediction")
st.markdown("Predict traffic speed using ML & DNN models.")

# User inputs
volume = st.number_input("Vehicle Count (Volume)", 0, 10000, 200)
temperature = st.number_input("Temperature (°C)", -10, 50, 25)
hour = st.slider("Hour of the Day", 0, 23, 8)
dayofweek = st.selectbox("Day of the Week (0=Monday, 6=Sunday)", list(range(7)), index=1)

model_choice = st.radio("Select Model", ["RandomForest", "MLP"])

if st.button("Predict Speed"):
    try:
        if model_choice == "RandomForest":
            speed = predict_speed_rf(volume, temperature, hour, dayofweek)
        else:
            speed = predict_speed_mlp(volume, temperature, hour, dayofweek)
        st.success(f"🚦 Predicted Traffic Speed: {speed:.2f} km/h")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Show Clustering PCA
st.subheader("Traffic Data Clustering")
base_dir = os.path.dirname(__file__)
clusters_path = os.path.join(base_dir, "models", "clusters_pca.png")
if os.path.exists(clusters_path):
    img = Image.open(clusters_path)
    st.image(img, caption="PCA + KMeans Clustering")

st.caption("Built by Team Aly — powered by scikit-learn & Streamlit")
