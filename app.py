import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model pipeline
pipeline = joblib.load("fraud_detection_pipeline2.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Fraud Detection in Financial Transactions")
st.markdown("Enter transaction details below to check if it's **fraudulent or legit**.")

type_map = {'CASH_OUT': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4}

# Input fields
#step = st.number_input("Step (transaction step/time)", min_value=1)
type_input = st.selectbox("Transaction Type", list(type_map.keys()))
amount = st.number_input("Amount", min_value=0.0, step=100.0)
oldbalanceOrg = st.number_input("Sender Balance Before Transaction", min_value=0.0, step=100.0)
newbalanceOrig = st.number_input("Sender Balance After Transaction", min_value=0.0, step=100.0)
oldbalanceDest = st.number_input("Receiver Balance Before Transaction", min_value=0.0, step=100.0)
newbalanceDest = st.number_input("Receiver Balance After Transaction", min_value=0.0, step=100.0)

# Predict
if st.button("Check Fraud"):
    x_input = pd.DataFrame([{
        #"step": step,
        "type": type_map[type_input],
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = pipeline.predict(x_input)[0]
    fraud_proba = pipeline.predict_proba(x_input)[0][1]
    threshold = 0.25

    if fraud_proba >= threshold:
        st.error(f"⚠️ Fraud Detected! ({fraud_proba:.2f})")
    else:
        st.success(f"✅ Legit Transaction ({fraud_proba:.2f})")

