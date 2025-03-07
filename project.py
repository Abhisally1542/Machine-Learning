import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image


# Load model and scaler
model = joblib.load('project.pkl')
scaler = joblib.load('scaler.pkl')

# User inputs
# Title and description
st.markdown("<h1 style='text-align: center; color: #FF5733;'>Machine Failure Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict if a Machine is Likely to Fail</h3>", unsafe_allow_html=True)
st.write("Enter the machine parameters below to check if it's at risk of failure.")

# Create two columns for a better layout
col1, col2 = st.columns(2)

# User inputs
with col1:
    temperature = st.number_input('Enter Temperature (°C):', min_value=-50.0, max_value=200.0, value=25.0, step=0.1)
    pressure = st.number_input('Enter Pressure (bar):', min_value=0.0, max_value=500.0, value=100.0, step=0.1)
    humidity = st.number_input('Enter Humidity (%):', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    vibration = st.number_input('Enter Vibration Level (Hz):', min_value=0.0, max_value=100.0, value=5.0, step=0.1)

with col2:
    load = st.number_input('Enter Load (%):', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    rpm = st.number_input('Enter RPM:', min_value=0, max_value=10000, value=1500, step=1)
    oil_quality = st.number_input('Enter Oil Quality Index (0-1):', min_value=0.0, max_value=1.0, value=0.8, step=0.01)


# Prediction
st.markdown("<br>", unsafe_allow_html=True)
if st.button('🔍 Predict', help="Click to predict machine failure risk"):
    try:
        input_data = np.array([[temperature, pressure, humidity, vibration, load, rpm, oil_quality]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        
        if prediction == 1:
            st.write('⚠️ Prediction: Machine is likely to FAIL')
        else:
            st.write('✅ Prediction: Machine is NOT likely to fail')
    except Exception as e:
        st.error(f'Error in prediction: {e}')
