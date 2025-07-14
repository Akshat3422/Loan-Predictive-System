import streamlit as slt
import pickle
import pandas as pd
import numpy as np
slt.title("loan predictive sysetm")
loan_dataset=pickle.load(open('loan_prediction_dataset.pkl','rb'))
loan_data=pd.DataFrame(loan_dataset)
loan_data.drop(['Loan_Status'],axis=1,inplace=True)
input_colums=loan_data.columns
slt.header("Enter Applicant Details")

Gender=slt.selectbox("What is your gender ?", ['Male', 'Female'])
Married=slt.selectbox("Are you Married ?", ['Yes', 'No'])
Dependents=slt.selectbox("Do you have any dependents?", ['0', '1', '2', '3+'])
Education=slt.selectbox("What is your education level?", ['Graduate', 'Not Graduate'])
Self_Employed=slt.selectbox("Are you self employed?", ['Yes', 'No'])
Credit_History=slt.selectbox("Do you have a credit history?", ['Yes', 'No'])
Property_Area=slt.selectbox("What is your property area?", ['Urban', 'Rural', 'Semiurban'])
New_ApplicantIncome=slt.number_input("What is your income?", min_value=0)
Loan_Amount_Term_in_months=slt.number_input("How long is your loan term( in months) ?", min_value=12, step=12)
CoapplicantIncome=slt.number_input("What is your co-applicant income?", min_value=0)
Has_CoapplicantIncome=slt.selectbox("Do you have a co-applicant?", ['Yes', 'No'])
LoanAmount=slt.number_input("What is your loan amount?", min_value=0)

input_dict = {
    'Gender': [1 if Gender == 'Male' else 0],
    'Married': [1 if Married == 'Yes' else 0],
    'Dependents': [0 if Dependents == '0' else 1 if Dependents == '1' else 2 if Dependents == '2' else 3],
    'Education': [1 if Education == 'Graduate' else 0],
    'Self_Employed': [1 if Self_Employed == 'Yes' else 0],
    'Credit_History': [1 if Credit_History == 'Yes' else 0],
    'Property_Area': [2 if Property_Area=='Urban' else 0 if Property_Area=='Rural' else 1],
    'Logged_New_ApplicantIncome': [np.log1p(New_ApplicantIncome)],
    'Loan_Amount_Term_in year': [Loan_Amount_Term_in_months/12],
    'Logged_CoapplicantIncome': [np.log1p(CoapplicantIncome)],
    'Has_CoapplicantIncome': [1 if Has_CoapplicantIncome=='Yes' else 0],
    'Logged_LoanAmount': [np.log1p(LoanAmount)]
    # Add other features in the same format
}
input_df=pd.DataFrame(input_dict)
model=pickle.load(open('loan_prediction_model.pkl','rb'))
# Prediction
if slt.button("Predict Loan Status"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0][1]
    slt.info(f"Probability of Loan Approval: {prediction_proba:.2%}")

    if prediction[0] == 1:
        slt.success(" Loan Approved!🎉🎊")
    else:
        slt.error(" Loan Not Approved🥺🥺")

