# 🏦 Bank Customer Churn Prediction

A comprehensive machine learning project focused on predicting bank customer churn using various classification algorithms. The goal is to identify customers likely to leave the bank to enable proactive retention strategies.

## 💻 Technologies and Libraries

| Category | Libraries/Tools |
| :--- | :--- |
| **Language** | Python |
| **Data Analysis** | `pandas`, `numpy` |
| **Visualization** | `seaborn`, `matplotlib` |
| **Machine Learning** | `scikit-learn` (Logistic Regression, SVC, KNN, Decision Tree, Random Forest, Gradient Boosting) |
| **Advanced ML** | `xgboost` |
| **Preprocessing** | `StandardScaler`, `One-Hot Encoding` |
| **Imbalance Handling** | `imbalanced-learn` (`SMOTE`) |
| **Model Persistence** | `joblib` |

## ✨ Key Features

*   **Exploratory Data Analysis (EDA):** Detailed analysis of the dataset structure, missing values, and statistical summaries.
*   **Data Preprocessing:** Handling of categorical features (`Geography`, `Gender`) using One-Hot Encoding and feature scaling using `StandardScaler`.
*   **Class Imbalance Handling:** Application of the **Synthetic Minority Over-sampling Technique (SMOTE)** to balance the `Exited` (churn) class.
*   **Model Comparison:** Evaluation of seven different classification models to determine the best predictor.
*   **Model Persistence:** Saving the best-performing model (`XGBoost` or `RandomForest`) using `joblib` for future deployment.

## 📋 Table of Contents

1.  [Project Goal](#project-goal)
2.  [Dataset](#dataset)
3.  [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
4.  [Model Training and Evaluation](#model-training-and-evaluation)
5.  [Results Summary](#results-summary)
6.  [Best Model](#best-model)
7.  [Getting Started](#getting-started)
8.  [Model Persistence](#model-persistence)
9.  [License](#license)
10. [Author](#author)

## 🎯 Project Goal

The primary objective of this project is to build a robust machine learning model that can accurately predict which bank customers are most likely to churn (leave the bank). This is a binary classification problem where the target variable `Exited` is 1 for churned customers and 0 otherwise.

## 📊 Dataset

The project uses the `Churn_Modelling.csv` dataset, which contains 10,000 records and 14 features.

| Feature | Data Type | Description |
| :--- | :--- | :--- |
| `RowNumber` | int64 | Row number (identifier, dropped) |
| `CustomerId` | int64 | Unique customer ID (identifier, dropped) |
| `Surname` | object | Customer's surname (identifier, dropped) |
| `CreditScore` | int64 | Credit score of the customer |
| `Geography` | object | Country of the customer (France, Spain, Germany) |
| `Gender` | object | Gender of the customer |
| `Age` | int64 | Age of the customer |
| `Tenure` | int64 | Number of years the customer has been with the bank |
| `Balance` | float64 | Account balance |
| `NumOfProducts` | int64 | Number of bank products the customer uses |
| `HasCrCard` | int64 | Whether the customer has a credit card (1=Yes, 0=No) |
| `IsActiveMember` | int64 | Whether the customer is an active member (1=Yes, 0=No) |
| `EstimatedSalary` | float64 | Estimated salary of the customer |
| **`Exited`** | int64 | **Target variable (1=Churn, 0=No Churn)** |

**Initial Class Distribution:**
The dataset exhibits a significant class imbalance:
*   **Not Exited (0):** 7963 customers
*   **Exited (1):** 2037 customers

## 🛠️ Data Preprocessing and Feature Engineering

1.  **Feature Removal:** Irrelevant identifier columns (`RowNumber`, `CustomerId`, `Surname`) were dropped.
2.  **One-Hot Encoding:** Categorical features (`Geography`, `Gender`) were converted into numerical format using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity.
3.  **Class Balancing:** The **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data to address the class imbalance, resulting in a balanced dataset for training:
    *   **Not Exited (0):** 7963
    *   **Exited (1):** 7963
4.  **Feature Scaling:** All features were scaled using `StandardScaler` to normalize the range of values, which is crucial for distance-based algorithms like SVC and KNN.

## 📈 Model Training and Evaluation

The preprocessed data was split into training and testing sets (`test_size=0.3`, `random_state=47`). Seven different classification models were trained and evaluated based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | **0.8548** | **0.8459** | **0.8596** | **0.8527** |
| XGBoost (XGB) | 0.8512 | 0.8436 | 0.8207 | 0.8253 |
| Support Vector Classifier (SVC) | 0.8273 | 0.8290 | 0.8151 | 0.8220 |
| Gradient Boosting Classifier (GBC) | 0.8301 | 0.8299 | 0.8207 | 0.8253 |
| K-Nearest Neighbors (KNN) | 0.8100 | 0.7966 | 0.8211 | 0.8087 |
| Decision Tree (DT) | 0.7966 | 0.7812 | 0.8113 | 0.7960 |
| Logistic Regression (LR) | 0.7625 | 0.7496 | 0.7724 | 0.7608 |

## 🏆 Best Model

The **Random Forest Classifier** achieved the highest overall performance metrics, particularly the highest **Accuracy (85.48%)** and **F1-Score (85.27%)**, making it the most suitable model for this prediction task.

## 🚀 Getting Started

Follow these steps to set up the project and run the analysis.

### Prerequisites

*   Python 3.x
*   The `Churn_Modelling.csv` dataset must be placed in the project root directory.

### Step 1: Clone the Repository (Conceptual)

Assuming this code is part of a repository:

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 2: Install Dependencies

All required libraries can be installed using `pip`:

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn xgboost joblib
```

### Step 3: Run the Analysis

The code snippets provided are typically run sequentially in a Jupyter Notebook or a Python script.

```python
# Example of key steps in the script
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
# ... (rest of the code)

# Load data
df = pd.read_csv('Churn_Modelling.csv')

# Preprocessing, SMOTE, Scaling, Train/Test Split...

# Train the best model (Random Forest)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Evaluate
# ...
```

## 💾 Model Persistence

The final XGBoost model was saved to disk using `joblib` for easy loading and deployment:

```python
import joblib
# Saving the model
joblib.dump(model_xgb, 'churn_predict_model')

# Loading the model
model = joblib.load('churn_predict_model')
```

The saved model file is named `churn_predict_model`.

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Hasnain006-nain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👨‍💻 Author

**Hasnain Haider**

