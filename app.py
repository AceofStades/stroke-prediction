import streamlit as st
import joblib

model = joblib.load('model-xgb.joblib')

st.title("Stroke Predictor")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
	gender = st.segmented_control("Gender", ["Male", "Female"], selection_mode="single", key="sex")
	if gender == "Male":
		gender = 1
	elif gender == "Female":
		gender = 0

with col2:
	ever_married = st.segmented_control("Ever Married?", ["Yes", "No"], selection_mode="single", key="marriage")
	if ever_married == "Yes":
		ever_married = 1
	elif ever_married == "No":
		ever_married = 0

with col3:
	hypertension = st.segmented_control("HyperTension", ["None", "Yes"], selection_mode="single", key="hypert")
	if hypertension == "Yes":
		hypertension = 1
	elif hypertension == "None":
		hypertension = 0


col4, col5, col6, = st.columns([1, 1, 1])

with col4:
	age = st.number_input("Age", min_value=0, max_value=130, value=18, step=1, placeholder="Enter your age", key="age")
with col5:
	bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=22.0, step=0.1, placeholder="Enter your BMI", key="bmi")
with col6:
	avg_glucose_level = st.number_input("Average glucose level", min_value=0.00, max_value=400.00, value=100.00, step=0.05, placeholder="Enter your Glucose Level", key="glucose")

col7, col8 = st.columns([3, 1])
with col7:
	work_type = st.segmented_control("Work Type: ", ["Government", "Never Worked", "Private", "Self Employed", "Child"], selection_mode="single", key="occupation")
with col8:
	heart_disease = st.segmented_control("Heart Disease", ["None", "Yes"], selection_mode="single", key="hd")

col9, col10 = st.columns([3, 1])
with col9:
	smoke = st.segmented_control("Smoking Status", ["Smokes", "Formerly Smokes", "Never", "Unknown"], selection_mode="single", key="smoke")
with col10:
	residency = st.segmented_control("Residency Type", ["Urban", "Rural"], selection_mode="single", key="residency")

st.write(gender, ever_married, hypertension, age, bmi, avg_glucose_level, work_type, heart_disease, smoke, residency)
