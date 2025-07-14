# Loan-Predictive-System
# 🏦 Loan Prediction System using Streamlit

A user-friendly web application built with **Streamlit** that predicts whether a loan will be approved based on applicant details like income, credit history, dependents, etc. The app uses a trained **machine learning model** stored as a `.pkl` file to make real-time predictions.

---

## 🚀 Features

- 📥 Collects user inputs through interactive UI (dropdowns, number fields)
- 🤖 Predicts **Loan Approval** status using trained ML model
- 📊 Displays **approval probability** as a confidence score
- 🔐 Log transformation for income/loan values to improve prediction accuracy
- 💼 Suitable for banks, NBFCs, or as a learning project

---

## 🧰 Tech Stack

| Component      | Technology                |
|----------------|----------------------------|
| Language       | Python                     |
| Interface      | Streamlit                  |
| Data Handling  | Pandas, NumPy              |
| Model Format   | Pickle (`.pkl`)            |
| ML Algorithm   | Logistic Regression        |

---

## 🏗️ Project Structure
loan-prediction-app/
│
├── app.py                        # Main Streamlit application
├── loan_prediction_model.pkl     # Trained ML model (Pickle)
├── loan_prediction_dataset.pkl   # Dataset used for training (Pickle format)
├── README.md                     # This documentation file



---

## 🧠 Features Collected from User

- Gender
- Marital Status
- Number of Dependents
- Education Level
- Employment Type
- Credit History
- Property Area
- Income (Applicant + Co-applicant)
- Loan Amount
- Loan Term Duration
- Co-applicant status

🔁 Some values are **log-transformed** (e.g., Income, Loan Amount) for better model performance.

---

## 🧪 Model Workflow

1. User fills in the applicant form via Streamlit interface.
2. Input is converted into model-compatible format using dictionary and `DataFrame`.
3. Features are passed into a pre-trained ML model (`loan_prediction_model.pkl`).
4. Model outputs:
   - Binary prediction → `Approved` or `Not Approved`
   - Confidence probability → e.g., `92.35%` approval chance

---

## ▶️ How to Run

### 1. Clone the Repository
```
git clone https://github.com/Akshat3422/Loan-Predictive-System.git
cd Loan-Predictive-System
```

### 2. Install Dependencies
pip install -r requirements.txt

### 3.Run the App
streamlit run app.py



