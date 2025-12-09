import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Loan Approval Prediction App")

st.write(
    "This app loads the final Logistic Regression model and predicts "
    "whether a loan application is likely to be approved."
)


@st.cache_resource
def load_model():
    with open("my_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.sidebar.header("Customer Information")


fico_score = st.sidebar.slider("FICO Score", min_value=300, max_value=850, value=700, step=5)
monthly_income = st.sidebar.number_input("Monthly Gross Income ($)", min_value=0, value=5000, step=500)
dti_ratio = st.sidebar.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

reason = st.sidebar.selectbox(
    "Loan Reason",
    ["credit_card_refinancing", "debt_consolidation", "home_improvement", "other"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["full_time", "part_time", "unemployed"]
)

ever_bankrupt = st.sidebar.selectbox(
    "Ever Bankrupt or Foreclosed?",
    ["no", "yes"]
)


def build_input_row():
   
    data = {}

   
    data["Monthly_Gross_Income"] = monthly_income
    data["Fico_Score"] = fico_score
    data["DTI_Ratio"] = dti_ratio

  
    # 'Reason_credit_card_refinancing', 'Reason_debt_consolidation',
    # 'Reason_home_improvement', 'Reason_other'
    data["Reason_credit_card_refinancing"] = 1 if reason == "credit_card_refinancing" else 0
    data["Reason_debt_consolidation"] = 1 if reason == "debt_consolidation" else 0
    data["Reason_home_improvement"] = 1 if reason == "home_improvement" else 0
    data["Reason_other"] = 1 if reason == "other" else 0


    data["Employment_Status_full_time"] = 1 if employment_status == "full_time" else 0
    data["Employment_Status_part_time"] = 1 if employment_status == "part_time" else 0
    data["Employment_Status_unemployed"] = 1 if employment_status == "unemployed" else 0

   
    data["Ever_Bankrupt_or_Foreclose"] = 1 if ever_bankrupt == "yes" else 0

    
    input_df = pd.DataFrame([data])
    return input_df

input_df = build_input_row()

st.subheader("Input Features (for debugging)")
st.dataframe(input_df)

if st.button("Predict Loan Approval"):
  
    proba_approved = model.predict_proba(input_df)[0, 1]
    st.write(f"Predicted probability of approval: **{proba_approved:.2%}**")

    if proba_approved >= 0.5:
        st.success("Prediction: The loan is **likely to be APPROVED**.")
    else:
        st.error("Prediction: The loan is **likely to be DENIED**.")
