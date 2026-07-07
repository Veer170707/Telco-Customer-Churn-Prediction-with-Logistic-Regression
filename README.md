# Telco Customer Churn Prediction with Logistic Regression

A machine learning pipeline that predicts customer churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), built with scikit-learn. The focus of this project is not just model accuracy, but a **correct, leakage-safe preprocessing pipeline** — a core skill for production ML systems.

## Overview

Customer churn (customers leaving a service) is a common business problem in subscription-based industries like telecom. This project builds a binary classifier to predict whether a customer will churn based on their account, service, and billing information.

## Dataset

- **Source:** Telco Customer Churn dataset (~7,000 customer records)
- **Target variable:** `Churn Value` (1 = churned, 0 = retained)
- **Features:** Demographics (gender, senior citizen status), account info (tenure, contract type, payment method), services subscribed (phone, internet, streaming, security add-ons), and billing (monthly charges, total charges)

## Approach

### 1. Data Cleaning
- Dropped irrelevant/redundant columns (`CustomerID`, `Zip Code`, `Lat Long`, `Count`, `Churn Reason`, `Churn Label`)
- Dropped constant columns with a single unique value (e.g. `Country`), which carry no predictive signal
- Dropped high-cardinality columns (e.g. `City`) to keep the feature space manageable for a linear model
- Converted `Total Charges` to numeric, coercing blank strings to `NaN`

### 2. Train/Validation/Test Split
- Split data **before** any fitting occurs: 60% train / 20% validation / 20% test
- This ensures no information from validation or test sets leaks into preprocessing

### 3. Handling Missing Values
- Missing `Total Charges` values imputed using the **median computed on the training set only**, then applied consistently to validation and test sets

### 4. Encoding Categorical Features
- Binary categorical columns (e.g. `Yes`/`No`, `Male`/`Female`) manually mapped to 0/1
- Multi-category columns (e.g. `Contract`, `Payment Method`, `Internet Service`) encoded using `OneHotEncoder`
- Encoder **fit only on training data**, then used to transform train, validation, and test sets — preventing category leakage and ensuring consistent columns across splits

### 5. Feature Scaling
- Continuous numeric features (e.g. `Tenure Months`, `Monthly Charges`, `Total Charges`) scaled using `StandardScaler`
- Scaler **fit only on training data**, then applied to validation and test sets

### 6. Modeling
- **Logistic Regression** (`liblinear` solver) trained on the fully preprocessed training set
- Evaluated on validation and held-out test sets to check generalization

## Why Leakage Prevention Matters

A recurring theme in this project is fitting every transformer (imputer, encoder, scaler) **only on the training set**, then reusing those fitted transformers on validation/test data. Fitting on the full dataset before splitting — a common beginner mistake — lets information about the test set influence preprocessing decisions, producing metrics that look better than the model will actually perform in the real world.

## Results

| Metric | Score |
|---|---|
| Train Accuracy |  0.92 |
| Validation Accuracy | 0.92 |
| Test Accuracy | 0.90 |

*(Run `logistic_project.py` to reproduce these numbers.)*

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (`OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `train_test_split`)

## How to Run

```bash
pip install pandas numpy scikit-learn
python logistic_project.py
```

Make sure `Telco_customer_churn.csv` is in the same directory as the script.

## Project Structure

```
├── Telco_customer_churn.csv
├── logistic_project.py
└── README.md
```

## Future Improvements

- Try tree-based models (Random Forest, XGBoost) for comparison
- Address class imbalance in the target variable (churned customers are typically a minority class) using class weights or resampling
- Add cross-validation for more robust performance estimates
- Add precision/recall/F1 and a confusion matrix, since accuracy alone can be misleading on imbalanced classification problems
