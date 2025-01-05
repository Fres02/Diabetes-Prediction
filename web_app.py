import pandas as pd
import numpy as np
import streamlit as st
import pickle

st.title("🏥 Diabetes Predictor")

st.write(
    "This app predicts the likelihood of diabetes based on patient details. "
    "Enter the required values in the sidebar and click **Submit** to see the result."
)

st.sidebar.header("📝 Enter Patient Details:")
Pregnancies = st.sidebar.number_input("Pregnancies:", min_value=0, step=1, value=0)
Glucose = st.sidebar.number_input("Glucose Level:", min_value=0.0, step=1.0, value=148.0)
BloodPressure = st.sidebar.number_input("Blood Pressure (mmHg):", min_value=0.0, step=1.0, value=72.0)
SkinThickness = st.sidebar.number_input("Skin Thickness (mm):", min_value=0.0, step=1.0, value=35.0)
Insulin = st.sidebar.number_input("Insulin Level:", min_value=0.0, step=1.0, value=0.0)
BMI = st.sidebar.number_input("BMI:", min_value=0.0, step=0.1, value=33.6)
DiabetesPedigreeFunction = st.sidebar.number_input(
    "Diabetes Pedigree Function:", min_value=0.0, step=0.01, value=0.627
)
Age = st.sidebar.number_input("Age:", min_value=0, step=1, value=50)

try:
    with open("Diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 The model file 'Diabetes_model.pkl' was not found. Please ensure the file is in the same directory as this script.")
    st.stop()

if st.sidebar.button("Submit"):
    x_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = model.predict(x_data)

    st.subheader("🩺 Prediction Result:")
    if prediction[0] == 1:
        st.markdown(
            '<div style="padding: 20px; background-color: #f8d7da; color: #842029; border-radius: 5px; text-align: center; font-weight: bold;">'
            "The person is likely to have diabetes."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding: 20px; background-color: #d1e7dd; color: #0f5132; border-radius: 5px; text-align: center; font-weight: bold;">'
            "The person is unlikely to have diabetes."
            "</div>",
            unsafe_allow_html=True,
        )
