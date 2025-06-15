import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model and scaler ===
MODEL_PATH = "D:/DM Project/calorie_model.pkl"
SCALER_PATH = "D:/DM Project/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="🔥 Calorie Predictor", layout="centered")
st.title("🔥 Calorie Burn Predictor")
st.markdown("Enter your workout and body details below to estimate the **calories burned**.")

# === Input form ===
with st.form("calorie_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        sex = st.radio("Sex", ["Male", "Female"])

    with col2:
        duration = st.slider("Duration of Workout (minutes)", 5, 180, 60)
        heart_rate = st.slider("Average Heart Rate", 60, 200, 120)
        body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

    submitted = st.form_submit_button("🔮 Predict")

# === Prediction logic ===
if submitted:
    # Encode and calculate features
    sex_encoded = 0 if sex == "Male" else 1
    bmi = weight / ((height / 100) ** 2 + 1e-8)
    hr_per_min = heart_rate / (duration + 1e-8)

    # Create DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp,
        'Sex': sex_encoded,
        'BMI': bmi,
        'HR_per_min': hr_per_min
    }])

    # Scale
    features_to_scale = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])

    # Predict
    y_pred_log = model.predict(input_data)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)

    st.success(f"🔥 Estimated Calories Burned: **{y_pred[0]:.2f}** kcal")
