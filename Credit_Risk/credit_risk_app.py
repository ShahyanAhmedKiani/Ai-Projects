import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.markdown("Predict the risk of loan default based on user information.")

# Load model & feature names
model = joblib.load("credit_risk_model.pkl")
features = joblib.load("feature_names.pkl")

# Input form
with st.form("input_form"):
    person_age = st.number_input("Age", min_value=18, max_value=100, help="Enter your age in years")
    person_income = st.number_input("Monthly Income", min_value=1000.0, help="Enter your total income")
    person_home_ownership = st.selectbox("Home Ownership", [0, 1, 2], format_func=lambda x: ["Rent", "Own", "Mortgage"][x])
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0, max_value=40)
    loan_intent = st.selectbox("Loan Purpose", [0, 1, 2, 3, 4, 5], format_func=lambda x: ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt Consolidation"][x])
    loan_grade = st.selectbox("Loan Grade", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: chr(65 + x))
    loan_amnt = st.number_input("Loan Amount", min_value=1000.0)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=1.0)
    cb_person_default_on_file = st.selectbox("Default on File", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=1, max_value=50)

    # Auto-calculated fields
    loan_percent_income = loan_amnt / (person_income + 1)
    debt_to_income_ratio = loan_amnt / (person_income + 1)

    submitted = st.form_submit_button("🔍 Predict")

# On Predict
if submitted:
    input_data = pd.DataFrame([[
        person_age, person_income, person_home_ownership, person_emp_length,
        loan_intent, loan_grade, loan_amnt, loan_int_rate,
        loan_percent_income, cb_person_default_on_file,
        cb_person_cred_hist_length, debt_to_income_ratio
    ]], columns=features)

    pred = model.predict(input_data)[0]
    score = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ High Risk of Default! (Risk Score: {score:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Risk Score: {score:.2f})")
