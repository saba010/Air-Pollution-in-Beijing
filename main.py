# main.py - Beijing Air Quality Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Set page
st.set_page_config(page_title="Beijing Air Predictor", page_icon="🌫️")

# Title
st.title("🌫️ Beijing Air Quality Predictor")
st.markdown("Predict PM2.5 pollution for any hour in Beijing")

# Load model (once, when app starts)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_tuned_model.pkl')
        return model
    except:
        st.error("Model file 'best_tuned_model.pkl' not found. Run training first.")
        return None

model = load_model()

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Hour slider
hour = st.sidebar.slider("Hour of day", 0, 23, 12, 
                         help="0 = midnight, 12 = noon, 23 = 11 PM")

# Weather inputs
st.sidebar.subheader("Weather Conditions")
temp = st.sidebar.number_input("Temperature (°C)", value=20.0, 
                               min_value=-20.0, max_value=45.0, step=1.0)
pres = st.sidebar.number_input("Pressure (hPa)", value=1013.0, 
                               min_value=980.0, max_value=1040.0, step=1.0)
wind = st.sidebar.number_input("Wind Speed (cumulated)", value=5.0, 
                               min_value=0.0, max_value=100.0, step=1.0)
dew = st.sidebar.number_input("Dew Point (°C)", value=10.0, 
                              min_value=-20.0, max_value=30.0, step=1.0)

# Time inputs
st.sidebar.subheader("Time Information")
month = st.sidebar.slider("Month", 1, 12, 6, 
                          help="1 = January, 6 = June, 12 = December")
is_weekend = st.sidebar.selectbox("Is weekend?", ["No", "Yes"])

# Historical pollution (simplified)
st.sidebar.subheader("📅 Recent Air Quality")

st.sidebar.markdown("""
*Typical Beijing values used:*
- **Yesterday:** 80 µg/m³ (average)
- **Last week:** 75 µg/m³ (average)
""")

# Hidden inputs with defaults (user doesn't see/change)
pm24_lag = 80.0
pm168_lag = 75.0

# Optional: Let advanced users change
with st.sidebar.expander("Advanced: Change historical values"):
    pm24_lag = st.number_input(
        "PM2.5 24h ago", 
        value=80.0, min_value=0.0, max_value=500.0
    )
    pm168_lag = st.number_input(
        "PM2.5 168h ago", 
        value=75.0, min_value=0.0, max_value=500.0
    )

# Calculate time features (same as in training)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * (month - 1) / 12)
month_cos = np.cos(2 * np.pi * (month - 1) / 12)
temp_minus_dew = temp - dew

# Time slot (morning/afternoon/evening/night)
if 5 <= hour < 12:
    time_slot = 0  # morning
elif 12 <= hour < 17:
    time_slot = 1  # afternoon
elif 17 <= hour < 21:
    time_slot = 2  # evening
else:
    time_slot = 3  # night

# Weekend flag
weekend_flag = 1 if is_weekend == "Yes" else 0

# Create input array in EXACT order model expects
input_data = [[
    hour_sin, hour_cos,
    month_sin, month_cos,
    temp_minus_dew,
    time_slot,
    temp, pres, wind,
    weekend_flag,
    pm24_lag, pm168_lag
]]

# Predict button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🌫️ Predict Air Quality", type="primary")

# Main area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Settings")
    st.write(f"**Time:** {hour}:00 (Month {month})")
    st.write(f"**Weather:** {temp}°C, {pres} hPa, Wind {wind}")
    st.write(f"**History:** Yesterday: {pm24_lag} µg/m³")
    st.write(f"**Weekend:** {is_weekend}")

# Make prediction when button clicked
if predict_button and model is not None:
    try:
        # Get prediction
        prediction = model.predict(input_data)[0]
        
        with col2:
            st.subheader("Prediction Result")
            
            # Big number display
            st.metric(label="Predicted PM2.5", value=f"{prediction:.1f} µg/m³")
            
            # Air quality category
            if prediction < 12:
                category = "Good"
                color = "green"
                emoji = "✅"
                advice = "Air quality is excellent. Perfect for outdoor activities."
            elif prediction < 35:
                category = "Moderate"
                color = "blue"
                emoji = "👍"
                advice = "Air quality is acceptable. Sensitive groups should consider reducing outdoor activity."
            elif prediction < 55:
                category = "Unhealthy for Sensitive Groups"
                color = "yellow"
                emoji = "⚠️"
                advice = "Children, elderly, and people with respiratory conditions should limit outdoor activity."
            elif prediction < 150:
                category = "Unhealthy"
                color = "orange"
                emoji = "😷"
                advice = "Everyone may experience health effects. Limit outdoor activity."
            else:
                category = "Hazardous"
                color = "red"
                emoji = "🚨"
                advice = "Health warning of emergency conditions. Avoid all outdoor activity."
            
            # Display category
            st.markdown(f"### {emoji} **{category}**")
            st.info(advice)
            
            # Additional info
            st.markdown("---")
            st.markdown("**Model Performance:**")
            st.write(f"- Average error: ±45.6 µg/m³")
            st.write(f"- Accuracy: 52.2% (R² score)")
            
            # Time of day tips
            st.markdown("**Best time for outdoor activities:**")
            if hour < 10 or hour > 18:
                st.write("⏰ Current hour is good (away from rush hours)")
            else:
                st.write("⏰ Consider early morning or evening (less traffic pollution)")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")

elif predict_button:
    st.error("Model not loaded. Check if 'best_tuned_model.pkl' exists.")

# Footer info
st.markdown("---")
st.markdown("### ℹ️ About This App")
st.write("""
This app predicts PM2.5 air pollution in Beijing using machine learning.

**How it works:**
1. Uses historical pollution data (yesterday, last week)
2. Considers weather (temperature, pressure, wind)
3. Accounts for time patterns (hour, month, weekday/weekend)
4. Predicts using a Random Forest model trained on Beijing data

**Model details:**
- MAE: 45.6 µg/m³ (average error)
- R²: 0.522 (explains 52% of variation)
- Features: 12 input parameters

**Data source:** Beijing Air Quality Dataset
""")

# Quick test button for debugging
with st.expander("🔧 Debug / Test"):
    if st.button("Show input features"):
        feature_names = [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "temp_minus_dew", "time_slot", "TEMP", "PRES", 
            "Iws", "is_weekend", "pm25_lag_24h", "pm25_lag_168h"
        ]
        df_debug = pd.DataFrame(input_data, columns=feature_names)
        st.dataframe(df_debug)
        
    if st.button("Test model loading"):
        try:
            test_model = joblib.load('best_tuned_model.pkl')
            st.success("Model loads successfully!")
        except Exception as e:
            st.error(f"Load failed: {e}")