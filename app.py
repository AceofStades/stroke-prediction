import streamlit as st
import joblib
import pandas as pd

def create_interaction_features(df):
    df_copy = df.copy()
    df_copy['age_x_bmi'] = df_copy['age'] * df_copy['bmi']
    df_copy['age_x_glucose'] = df_copy['age'] * df_copy['avg_glucose_level']
    df_copy['bmi_x_glucose'] = df_copy['bmi'] * df_copy['avg_glucose_level']
    df_copy['risk_factor_count'] = df_copy['hypertension'] + df_copy['heart_disease']
    return df_copy


try:
    pipeline = joblib.load('model-lr-feature-eng.joblib')
except FileNotFoundError:
    st.error("Model file not found. Make sure 'model-lr-feature-eng.joblib' is present.")
    st.stop()


st.title("Stroke Prediction Engine 🩺")
st.write("This app predicts the risk of stroke based on your health metrics. Please provide your information below.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    hypertension = st.radio("Do you have hypertension?", ["Yes", "No"], horizontal=True)
    heart_disease = st.radio("Do you have a heart disease?", ["Yes", "No"], horizontal=True)

with col2:
    ever_married = st.selectbox("Have you ever been married?", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Child", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

st.divider()
st.subheader("Health Measurements")
col3, col4 = st.columns(2)
with col3:
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=120.0, step=0.1)
with col4:
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=100.0, value=28.5, step=0.1)


if st.button("Predict Stroke Risk", type="primary"):

    gender_num = 1 if gender == "Male" else 0
    hypertension_num = 1 if hypertension == "Yes" else 0
    heart_disease_num = 1 if heart_disease == "Yes" else 0
    ever_married_num = 1 if ever_married == "Yes" else 0
    residence_num = 1 if Residence_type == "Urban" else 0

    work_Private = 1 if work_type == 'Private' else 0
    work_Self_employed = 1 if work_type == 'Self-employed' else 0
    work_Govt_job = 1 if work_type == 'Govt_job' else 0
    work_children = 1 if work_type == 'Child' else 0
    work_Never_worked = 1 if work_type == 'Never_worked' else 0

    smoke_formerly = 1 if smoking_status == 'formerly smoked' else 0
    smoke_never = 1 if smoking_status == 'never smoked' else 0
    smoke_smokes = 1 if smoking_status == 'smokes' else 0
    smoke_Unknown = 1 if smoking_status == 'Unknown' else 0

    # Create a DataFrame from the inputs.
    input_data = pd.DataFrame({
        'gender': [gender_num],
        'age': [age],
        'hypertension': [hypertension_num],
        'heart_disease': [heart_disease_num],
        'ever_married': [ever_married_num],
        'Residence_type': [residence_num],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'work_type_Govt_job': [work_Govt_job],
        'work_type_Never_worked': [work_Never_worked],
        'work_type_Private': [work_Private],
        'work_type_Self-employed': [work_Self_employed],
        'work_type_children': [work_children],
        'smoking_status_Unknown': [smoke_Unknown],
        'smoking_status_formerly smoked': [smoke_formerly],
        'smoking_status_never smoked': [smoke_never],
        'smoking_status_smokes': [smoke_smokes]
    })

    prediction = pipeline.predict(input_data)
    prediction_proba = pipeline.predict_proba(input_data)

    st.subheader("Prediction Result")
    probability_of_stroke = prediction_proba[0][1]

    if prediction[0] == 1:
        st.error(f"High Risk of Stroke (Probability: {probability_of_stroke*100:.2f}%)", icon="🚨")
        st.warning("This is a predictive model and not a substitute for professional medical advice. Please consult a doctor.", icon="⚠️")
    else:
        st.success(f"Low Risk of Stroke (Probability: {probability_of_stroke*100:.2f}%)", icon="✅")
        st.info("Continue to maintain a healthy lifestyle. Regular check-ups are always recommended.", icon="ℹ️")
