# CUSTOMER-CHURN-PREDICTION

A machine learning project that predicts customer churn for a bank using various classification algorithms. The model helps identify customers who are likely to leave the bank, enabling proactive retention strategies.

## Overview

This project analyzes customer data from a bank to predict whether a customer will churn (leave the bank) or not. Multiple machine learning models are trained and compared to find the best performing algorithm.

## Dataset

The dataset (`Churn_Modelling.csv`) contains 10,000 customer records with the following features:

- **CreditScore**: Customer's credit score
- **Geography**: Customer's location (France, Spain, Germany)
- **Gender**: Male or Female
- **Age**: Customer's age
- **Tenure**: Number of years with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products used
- **HasCrCard**: Whether customer has a credit card (1/0)
- **IsActiveMember**: Whether customer is an active member (1/0)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Target variable (1 = churned, 0 = retained)

## Project Workflow

### 1. Data Preprocessing
- Removed irrelevant columns (RowNumber, CustomerId, Surname)
- Handled categorical variables using one-hot encoding
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset
- Standardized features using StandardScaler

### 2. Models Implemented

Seven classification algorithms were trained and evaluated:

1. **Logistic Regression (LR)**
2. **Support Vector Classifier (SVC)**
3. **K-Nearest Neighbors (KNN)**
4. **Decision Tree (DT)**
5. **Random Forest (RF)**
6. **Gradient Boosting Classifier (GBC)**
7. **XGBoost (XGB)**

### 3. Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 76.25% | 74.96% | 77.24% | 76.08% |
| SVC | 82.73% | 82.90% | 81.51% | 82.20% |
| KNN | 80.99% | 79.66% | 82.11% | 80.87% |
| Decision Tree | 79.66% | 78.12% | 81.13% | 79.60% |
| **Random Forest** | **85.48%** | **84.59%** | **85.96%** | **85.27%** |
| Gradient Boosting | 83.01% | 82.99% | 82.07% | 82.53% |
| XGBoost | 85.12% | 84.36% | - | - |

**Best Model**: Random Forest with 85.48% accuracy

## Installation

# Clone the repository

git clone https://github.com/Hasnain006-nain/CUSTOMER-CHURN-PREDICTION.git
cd CUSTOMER-CHURN-PREDICTION

# Install required packages

pip install pandas numpy seaborn scikit-learn imbalanced-learn xgboost joblib

- Requirements
- pandas
- numpy
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- joblib
- Usage

---- Ensure Churn_Modelling.csv is in the project directory

---- Run the Jupyter notebook Untitled1.ipynb

---- The trained model is saved as churn_predict_model using joblib
     Loading the Saved Model

---- import joblib
     model = joblib.load('churn_predict_model')

# Make predictions
 predictions = model.predict(new_data)

# Key Insights

- The dataset was imbalanced (7963 retained vs 2037 churned),requiring SMOTE for balancing
- Random Forest achieved the best overall performance
- Feature engineering and standardization significantly improved model accuracy
- Ensemble methods (RF, GBC, XGB) outperformed traditional algorithms

# Project Structure

CUSTOMER-CHURN-PREDICTION/
│
├── Churn_Modelling.csv          # Dataset
├── Untitled1.ipynb              # Main notebook
├── churn_predict_model          # Saved XGBoost model
└── README.md                    # Project documentation


# Future Improvements

- Hyperparameter tuning for better performance
- Feature importance analysis
- Cross-validation for robust evaluation
- Deployment as a web application
- Real-time prediction API

# Author
Hasnian Haider

# License
This project is open source and available for educational purposes.


---
