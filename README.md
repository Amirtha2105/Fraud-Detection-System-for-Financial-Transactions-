# 🛡️ Fraud Detection System for Financial Transactions

A Machine Learning-based solution to detect fraudulent financial transactions in real-time. Built as part of a 1-month development sprint to explore end-to-end ML workflows.

## Problem Statement

Financial institutions must proactively detect fraudulent activities in massive streams of transactions. The aim of this project is to build a robust classification model using historical transaction data and deploy it with a simple simulation dashboard.

## Tech Stack

- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Visualization & Deployment**: Streamlit  
- **IDE**: Jupyter Notebook  

## Project Timeline (4 Weeks)

### Week 1 - Data Ingestion & EDA
- Loaded data from Kaggle or simulated CSV
- Exploratory Data Analysis (EDA) on fraud vs non-fraud patterns
- Dealt with class imbalance using SMOTE & undersampling

### Week 2 - Feature Engineering & Modeling
- Created features: time gaps, transaction amounts, user behavior trends
- Trained ML models: Logistic Regression, Decision Tree, XGBoost
- Evaluated with metrics: F1 Score, AUC-ROC, Precision-Recall

### Week 3 - Optimization & Pipeline Building
- Hyperparameter tuning using `GridSearchCV`
- Built a testable ML pipeline for future live inputs

### Week 4 - Final Model & Dashboard
- Finalized and saved the best model
- Deployed a simulation dashboard using Streamlit
- Documented findings, model limitations, and improvements

## Outputs

- ML model that classifies transactions as fraud/non-fraud
- Model pipeline ready for real-time testing
- Streamlit dashboard for simulation & demo

## Future Improvements

- Use deep learning for sequential fraud pattern detection  
- Integrate real-time data via APIs  
- Implement advanced anomaly detection techniques

## 📂 Folder Structure

├── fraud2/ # Jupyter notebook for EDA, feature engineering, modeling and deployment

├── fraud_detection_pipeline2.pkl/ # Saved ML models (.pkl)

├── app/ # Streamlit dashboard files

└── README.md # Project overview
