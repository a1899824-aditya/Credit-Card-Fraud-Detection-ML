# Credit-Card-Fraud-Detection-ML
Credit card fraud detection using feature selection, multiple ML models, model evaluation, and a Streamlit web app for real-time predictions.


# Credit Card Fraud Detection – Machine Learning Project

This project builds an end-to-end machine learning pipeline to detect fraudulent credit card transactions using multiple classification algorithms and feature selection techniques.

The system includes model training, feature engineering, evaluation, model comparison, and deployment through an interactive Streamlit web application.

---

## Project Objectives

• Detect fraudulent credit card transactions  
• Compare multiple machine learning models  
• Apply feature selection techniques  
• Deploy the final model using Streamlit  
• Generate explainable predictions using feature importance and SHAP values  

---

## Dataset

The dataset contains anonymized credit card transactions with features V1–V28 generated through PCA transformation, along with transaction time and amount.

Key fields:

• Time – seconds elapsed between transactions  
• Amount – transaction value  
• V1 – V28 – anonymized PCA features  
• Class – fraud label (1 = Fraud, 0 = Not Fraud)

---

## Feature Engineering

Additional features created:

• Hour extracted from Time  
• Scaled transaction amount  

Feature selection methods used:

• Mutual Information  
• Chi-Square  
• Recursive Feature Elimination (RFE)

---

## Models Implemented

The following machine learning models were trained and compared:

• Random Forest  
• Logistic Regression  
• XGBoost  

Evaluation metrics used:

• Accuracy  
• Precision  
• Recall  
• F1 Score (Fraud Class)  
• ROC AUC  

---

## Model Deployment

A Streamlit web application allows users to:

• Upload transaction datasets  
• Select prediction models  
• Generate fraud predictions  
• View fraud probabilities  
• Export prediction results  

Outputs generated:

• predictions.csv  
• feature_importance.csv  
• shap_values.csv  

---

## Running the Application

Install dependencies:


pip install streamlit pandas scikit-learn xgboost shap joblib


Run the application:



The application will open in your browser.

---

## Technologies Used

Python  
Scikit-Learn  
XGBoost  
Streamlit  
SHAP  
Pandas  
Matplotlib  

---

## Author

Aditya Venugopalan  
Master of Data Science – University of Adelaide
