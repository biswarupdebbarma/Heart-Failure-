import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessing objects from the saved artifact
loaded_artifact = joblib.load("heart_failure_artifacts.pkl")
loaded_model = loaded_artifact["model"]
loaded_scaler = loaded_artifact["scaler"]
feature_names = loaded_artifact["feature_names"]

st.title("Heart Failure Prediction App by Biswarup Debbarma")

st.write("Please enter the patient's details below:")

# Create input fields for each feature
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
anemia = st.selectbox("Anemia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, value=100)
diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100, value=50)
high_blood_pressure = st.selectbox("High Blood Pressure", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
platelets = st.number_input("Platelets", min_value=0, value=250000)
serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium", min_value=0, value=135)
smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes") 
follow_up_months = st.number_input("Follow-up Months", min_value=0, value=12)

if st.button("Predict"):
	patient_data = pd.DataFrame([{
		"age": age,
		"anaemia": anemia,
		"creatinine_phosphokinase": creatinine_phosphokinase,
		"diabetes": diabetes,
		"ejection_fraction": ejection_fraction,
		"high_blood_pressure": high_blood_pressure,
		"platelets": platelets,
		"serum_creatinine": serum_creatinine,
		"serum_sodium": serum_sodium,
		"sex": sex,
		"smoking": smoking,
		"time": follow_up_months,
	}], columns=feature_names)

	patient_data_scaled = loaded_scaler.transform(patient_data)
	prediction = loaded_model.predict(patient_data_scaled)[0]
	probability = loaded_model.predict_proba(patient_data_scaled)[0, 1]

	if prediction == 1:
		st.error(f"Prediction: Higher risk of heart failure ({probability:.1%} probability)")
	else:
		st.success(f"Prediction: Lower risk of heart failure ({probability:.1%} probability)")

