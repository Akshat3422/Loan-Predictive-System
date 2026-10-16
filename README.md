# Loan Prediction System

A Streamlit-based machine learning application that predicts whether a loan application is likely to be approved based on applicant details such as income, marital status, credit history, dependents, and loan amount.

The project loads a trained model from a pickle file and provides an interactive form for users to enter applicant information and receive a prediction along with a confidence score.

---

## Overview

This repository contains a complete end-to-end loan approval prediction workflow:

- A clean and interactive front-end built with Streamlit
- A trained classification model saved as a pickle artifact
- A dataset used to prepare and validate the input feature structure
- A user-friendly prediction interface for quick decision support

The app is designed for learning, demo, and experimentation purposes and can be adapted for real-world financial use cases.

---

## Features

- Interactive form to collect applicant details
- Predicts loan approval status (`Approved` / `Not Approved`)
- Displays probability/confidence score for the result
- Handles both categorical and numerical inputs
- Uses log transformations for income and loan-related values to improve model behavior
- Simple deployment with a single Streamlit command

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| ML Libraries | scikit-learn, xgboost |
| Model Serialization | Pickle (.pkl) |
| Data Source | CSV dataset |

---

## Project Structure

```text
Loan-Predictive-System/
├── app.py                           # Streamlit application
├── Dataset.csv                     # Loan application dataset
├── data.ipynb                      # Notebook for exploration / preprocessing
├── loan_prediction_model.pkl        # Serialized trained ML model
├── loan_prediction_dataset.pkl      # Dataset used to build feature structure
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore settings (if present)
```

---

## Dataset

The project uses a loan dataset containing applicant and financial attributes such as:

- Gender
- Marital status
- Dependents
- Education level
- Employment status
- Credit history
- Property area
- Applicant income
- Co-applicant income
- Loan amount
- Loan term duration
- Loan status (target variable)

The dataset is stored in `Dataset.csv` and is used to generate the feature schema expected by the trained model.

---

## How the App Works

1. The user opens the app in the browser.
2. They fill in the applicant details using dropdowns and numeric input fields.
3. The application converts the values into a DataFrame matching the model's expected feature names.
4. Selected numerical values are transformed using log scaling (`np.log1p`) to improve consistency with training data.
5. The model predicts whether the loan is approved.
6. The app displays:
   - approval result
   - prediction probability

---

## Model Behavior

The app loads a pre-trained model from:

- `loan_prediction_model.pkl`

It also loads the reference dataset from:

- `loan_prediction_dataset.pkl`

This ensures the app builds the same feature columns as the training stage and passes the correct input structure to the model.

---

## Input Fields

The UI captures the following information:

- Gender
- Married
- Dependents
- Education
- Self-employed status
- Credit history
- Property area
- Applicant income
- Co-applicant income
- Loan amount
- Loan term in months
- Co-applicant presence

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Akshat3422/Loan-Predictive-System.git
cd Loan-Predictive-System
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Example Output

After entering user information and clicking the prediction button, the app will display:

- a probability value such as `92.35%`
- a final message such as `Loan Approved!` or `Loan Not Approved`

---

## Requirements

The project dependencies are listed in `requirements.txt` and include:

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
```

---

## Notes

- The app relies on preprocessed feature names and encoding logic matching the training pipeline.
- Some values, especially income and loan amount, are log-transformed before prediction.
- This project is a strong example of a lightweight ML deployment using Streamlit for rapid prototyping and demonstration.

---

## License

This project does not appear to include a specific license file in the repository. If you plan to share or use it publicly, it is recommended to add an appropriate open-source license.

---

## Contact / Project Origin

This project was created for loan prediction experimentation and demonstration using a machine learning model to estimate approval likelihood based on applicant profile data.
