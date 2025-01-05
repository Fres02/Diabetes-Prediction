import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Diabetes Predictor App")

Pregnancies	= float(st.text_input("Enter the value for Pregnancies","6"))
Glucose	= float(st.text_input("Enter the value for Glucose","148"))
BloodPressure	= float(st.text_input("Enter the value for BloodPressure","72"))
SkinThickness	= float(st.text_input("Enter the value for SkinThickness","35"))
Insulin	= float(st.text_input("Enter the value for Insulin","0"))
BMI	= float(st.text_input("Enter the value for BMI","33.6"))
DiabetesPedigreeFunction = float(st.text_input("Enter the value for DiabetesPedigreeFunction","0.627"))
Age= float(st.text_input("Enter the value for Age","50"))

import pickle

with open('Diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)


x_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
x_data = np.array(x_data)
predictions = model.predict(x_data)

st.subheader('Do the person have Diabetes?')
if predictions[0] == 1:
    st.write("YES")
else:
    st.write("NO")