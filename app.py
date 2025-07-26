# streamlit_app.py

import streamlit as st
import numpy as np
import pickle

# 🔹 Load the model and scaler
with open("scaler_voting_model_credit_risk.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("voting_model_credit_risk.pkl", "rb") as f:
    model = pickle.load(f)

# 🔹 App title
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("💳 Credit Risk Classifier")
st.markdown("Predict whether a loan will be approved or rejected based on applicant details.")

# 🔹 Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
        LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
        Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
        Credit_History = st.selectbox("Credit History", [1.0, 0.0])
        Gender_Male = st.selectbox("Gender", ['Male', 'Female']) == 'Male'
        Self_Employed_Yes = st.selectbox("Self Employed", ['Yes', 'No']) == 'Yes'

    with col2:
        Married_Yes = st.selectbox("Married", ['Yes', 'No']) == 'Yes'
        Education_Not_Graduate = st.selectbox("Education", ['Graduate', 'Not Graduate']) == 'Not Graduate'
        Property_Area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
        Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])

    submit = st.form_submit_button("🔍 Predict")

# 🔹 On Submit
if submit:
    TotalIncome = ApplicantIncome + CoapplicantIncome

    # One-hot encoding for Property Area
    Property_Area_Semiurban = 1 if Property_Area == 'Semiurban' else 0
    Property_Area_Urban = 1 if Property_Area == 'Urban' else 0

    # One-hot encoding for Dependents
    Dependents_1 = 1 if Dependents == '1' else 0
    Dependents_2 = 1 if Dependents == '2' else 0
    Dependents_3_plus = 1 if Dependents == '3+' else 0

    # Feature vector as per training order
    input_data = np.array([[
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        TotalIncome,
        int(Gender_Male),
        Property_Area_Semiurban,
        Property_Area_Urban,
        int(Self_Employed_Yes),
        int(Married_Yes),
        int(Education_Not_Graduate),
        Dependents_1,
        Dependents_2,
        Dependents_3_plus
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    proba = model.predict_proba(input_scaled)[0]
    prediction = np.argmax(proba)  # 0 or 1

    confidence = round(proba[prediction] * 100, 2)

    if prediction == 1:
        st.success(f"✅ Loan Approved with {confidence}% confidence")
    else:
        st.error(f"❌ Loan Rejected with {confidence}% confidence")
