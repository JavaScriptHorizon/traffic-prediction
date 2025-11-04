import streamlit as st
from src.predict import predict_speed

# Page configuration
st.set_page_config(page_title="Traffic Speed Predictor", page_icon="🚦")

st.title("🚗 Traffic Speed Prediction")
st.markdown("Predict the average traffic speed based on traffic conditions.")

# User inputs
volume = st.number_input("Vehicle Count (Volume)", min_value=0, value=200)
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=25)
hour = st.slider("Hour of the Day", 0, 23, 8)
dayofweek = st.selectbox("Day of the Week (0=Monday, 6=Sunday)", list(range(7)), index=1)

# Prediction button
if st.button("Predict Speed"):
    try:
        speed = predict_speed(volume, temperature, hour, dayofweek)
        st.success(f"🚦 Predicted Traffic Speed: {speed:.2f} km/h")
    except Exception as e:
        st.error(f"❌ Error: {e}")

st.caption("Built by Mohamed Walid — powered by scikit-learn & Streamlit")
