# 🎯 Customer Churn Prediction

A machine learning project that predicts customer churn for banking institutions using advanced classification algorithms. This system analyzes customer behavior patterns and identifies customers at risk of leaving, enabling proactive retention strategies and reducing customer attrition rates.

## ✨ Features

- **Multiple ML Models**: Implements 7 different classification algorithms for comparison
- **Imbalanced Data Handling**: Uses SMOTE technique to balance the dataset
- **Feature Engineering**: Automated preprocessing and encoding of categorical variables
- **Model Persistence**: Saves trained models for instant deployment
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, recall, and F1-score
- **Visual Analytics**: Seaborn visualizations for model comparison
- **Scalable Pipeline**: StandardScaler for feature normalization
- **Production Ready**: Exportable model for real-world deployment

## Topics

`machine-learning` `python` `data-science` `classification` `churn-prediction` `xgboost` `random-forest` `scikit-learn` `banking` `customer-analytics`

## 📋 Table of Contents

- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Model Performance](#-model-performance)
- [Configuration](#️-configuration)
- [Dataset Details](#-dataset-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🎬 Demo

The system provides:
- **Accuracy Comparison**: Visual bar plots comparing all 7 models
- **Precision Analysis**: Detailed precision metrics for each algorithm
- **Best Model Selection**: Random Forest achieves 85.48% accuracy
- **Saved Model**: Pre-trained XGBoost model ready for predictions

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 4GB RAM minimum
- Windows/Linux/MacOS


### Step 1: Clone the Repository
git clone https://github.com/Hasnain006-nain/CUSTOMER-CHURN-PREDICTION.git
cd CUSTOMER-CHURN-PREDICTION

# Step 2: Install Dependencies
pip install -r requirements.txt

# Step 3: Verify Dataset
Ensure Churn_Modelling.csv is in the project directory with 10,000 customer records.

# 📖 Usage Running the Jupyter Notebook
jupyter notebook Untitled1.ipynb

# Loading the Pre-trained Model
import joblib
import numpy as np

# Load the saved model
model = joblib.load('churn_predict_model')

# Prepare your data (11 features after preprocessing)
# Features: 
           CreditScore, Age, Tenure, Balance, NumOfProducts, 
           HasCrCard, IsActiveMember, EstimatedSalary,
           Geography_Germany, Geography_Spain, Gender_Male

sample_data = np.array([[619, 42, 2, 0.0, 1, 1, 1, 101348.88, 0, 0, 0]])

# Make prediction
prediction = model.predict(sample_data)
print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")


# The pipeline includes:

 1. Data loading and exploration
 2. Preprocessing and feature engineering
 3. SMOTE balancing
 4. Model training (7 algorithms)
 5. Performance evaluation
 6. Model saving

# 📁 Project Structure
CUSTOMER-CHURN-PREDICTION/
│
├── Churn_Modelling.csv          # Dataset with 10,000 customer records
├── Untitled1.ipynb              # Main Jupyter notebook
├── churn_predict_model          # Saved XGBoost model
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE                      # MIT License

# 🔧 How It Works
1. Data Loading and Exploration

df = pd.read_csv('Churn_Modelling.csv')
df.info()  # 10,000 entries, 14 columns
df.isnull().sum()  # No missing values

Purpose: Load and understand the dataset structure
Why: Ensures data quality and identifies preprocessing needs
Output: 10,000 customer records with 14 features

# 2. Data Preprocessing

# Remove irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)
Purpose: Clean and transform data for machine learning
Why:

RowNumber, CustomerId, Surname don't contribute to predictions
ML algorithms require numerical inputs
One-hot encoding prevents ordinal assumptions
Result: 11 numerical features ready for modeling

# 3. Handling Imbalanced Data with SMOTE

from imblearn.over_sampling import SMOTE

X = df.drop('Exited', axis=1)
y = df['Exited']

X_res, y_res = SMOTE().fit_resample(X, y)
Purpose: Balance the dataset (7963 retained vs 2037 churned)
Why: Imbalanced data causes models to favor the majority class
Technology: SMOTE (Synthetic Minority Over-sampling Technique)

How SMOTE Works:

Identifies minority class samples (churned customers)
Finds k-nearest neighbors for each minority sample
Creates synthetic samples along the line segments
Balances dataset to 7963 vs 7963
Impact: Prevents model bias, improves recall for churned customers

# 4. Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Purpose: Normalize features to same scale
Why:

Features have different ranges (Age: 18-92, Balance: 0-250,898)
Distance-based algorithms (SVC, KNN) are sensitive to scale
Improves convergence speed for gradient-based models
Formula: z = (x - μ) / σ
Where μ = mean, σ = standard deviation

# 5. Model Training

Logistic Regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)

- Type: Linear classifier
- Best For: Baseline model, interpretable coefficients
- Accuracy: 76.25%

Support Vector Classifier (SVC)

from sklearn import svm
svm = svm.SVC()
svm.fit(X_train, y_train)

- Type: Kernel-based classifier
- Best For: Non-linear decision boundaries
- Accuracy: 82.73%

K-Nearest Neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

- Type: Instance-based learning
- Best For: Simple, no training phase
- Accuracy: 80.99%

Decision Tree

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

- Type: Tree-based classifier
- Best For: Interpretable rules, handles non-linearity
- Accuracy: 79.66%

Random Forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

- Type: Ensemble of decision trees
- Best For: High accuracy, reduces overfitting
- Accuracy: 85.48% ⭐ BEST

Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

- Type: Sequential ensemble
- Best For: Corrects previous model errors
- Accuracy: 83.01%

XGBoost

import xgboost as xgb
model_xgb = xgb.XGBClassifier(random_state=42, verbosity=0)
model_xgb.fit(X_train, y_train)

- Type: Optimized gradient boosting
- Best For: Production deployment, speed
- Accuracy: 85.12%

# 6. Model Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

Metrics Explained:

Accuracy: Overall correctness = (TP + TN) / Total
Precision: Of predicted churns, how many actually churned = TP / (TP + FP)
Recall: Of actual churns, how many we caught = TP / (TP + FN)
F1-Score: Harmonic mean of precision and recall = 2 × (P × R) / (P + R)
Why Multiple Metrics:

Accuracy alone can be misleading with imbalanced data
Precision matters for targeted retention campaigns (avoid false alarms)
Recall matters for not missing at-risk customers
F1-Score balances both concerns

# 7. Model Persistence

import joblib
joblib.dump(model_xgb, 'churn_predict_model')
Purpose: Save trained model for deployment
Why: Avoid retraining, enable production use
Format: Pickle-based serialization via joblib

# 📊 Model Performance

Accuracy Comparison
| Model | Accuracy | Precision | Recall | F1-Score | |-------|----------|-----------|--------|----------| | Logistic Regression | 76.25% | 74.96% | 77.24% | 76.08% | | Support Vector Classifier | 82.73% | 82.90% | 81.51% | 82.20% | | K-Nearest Neighbors | 80.99% | 79.66% | 82.11% | 80.87% | | Decision Tree | 79.66% | 78.12% | 81.13% | 79.60% | | Random Forest | 85.48% | 84.59% | 85.96% | 85.27% | | Gradient Boosting | 83.01% | 82.99% | 82.07% | 82.53% | | XGBoost | 85.12% | 84.36% | 82.07% | 82.53% |

# Key Insights
✅ Best Overall Model: Random Forest (85.48% accuracy)
✅ Best Precision: SVC (82.90%)
✅ Best Recall: Random Forest (85.96%)
✅ Fastest Training: Logistic Regression
✅ Production Choice: XGBoost (speed + accuracy balance)

# Performance Visualization
The notebook includes bar plots comparing:

Model accuracy across all 7 algorithms
Precision scores for each model
Visual identification of best performers


# ⚙️ Configuration

Adjusting Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, 
    test_size=0.3,  # 70% train, 30% test
    random_state=47  # For reproducibility
)
Recommendations:

test_size=0.2: More training data, less validation
test_size=0.3: Balanced (current setting)
test_size=0.4: More robust testing, less training data
SMOTE Parameters
X_res, y_res = SMOTE(
    sampling_strategy='auto',  # Balance to 50-50
    k_neighbors=5,  # Number of neighbors for synthesis
    random_state=42
).fit_resample(X, y)
XGBoost Hyperparameters
model_xgb = xgb.XGBClassifier(
    random_state=42,
    verbosity=0,
    n_estimators=100,  # Number of trees
    max_depth=6,  # Tree depth
    learning_rate=0.3,  # Step size
    subsample=1.0  # Fraction of samples per tree
)

# 📊 Dataset Details
Source
File: Churn_Modelling.csv
Records: 10,000 customers
Features: 14 columns (11 after preprocessing)
Target: Binary classification (Exited: 0 or 1)
Feature Descriptions
| Feature | Type | Description | Range/Values | |---------|------|-------------|--------------| | CreditScore | Numerical | Customer's credit score | 350-850 | | Geography | Categorical | Customer location | France, Spain, Germany | | Gender | Categorical | Customer gender | Male, Female | | Age | Numerical | Customer age | 18-92 years | | Tenure | Numerical | Years with bank | 0-10 years | | Balance | Numerical | Account balance | 0-250,898 | | NumOfProducts | Numerical | Number of products | 1-4 | | HasCrCard | Binary | Has credit card | 0 (No), 1 (Yes) | | IsActiveMember | Binary | Active membership | 0 (No), 1 (Yes) | | EstimatedSalary | Numerical | Estimated salary | 11-199,992 | | Exited | Binary | Churned (Target) | 0 (No), 1 (Yes) |

Class Distribution

Retained Customers: 7,963 (79.63%)
Churned Customers: 2,037 (20.37%)
Imbalance Ratio: 3.9:1
After SMOTE Balancing
Retained: 7,963 (50%)
Churned: 7,963 (50%)
Total Samples: 15,926
🛠️ Troubleshooting
Issue: ImportError for imblearn
Solution:

pip install imbalanced-learn
Issue: XGBoost installation fails
Solution for Windows:

pip install xgboost
If fails, try:
conda install -c conda-forge xgboost
Solution for Linux/Mac:

pip install xgboost
Issue: Low model accuracy
Solutions:

Increase training data: Collect more customer records
Feature engineering: Create new features (e.g., Balance/Salary ratio)
Hyperparameter tuning: Use GridSearchCV
Try deep learning: Neural networks for complex patterns
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
Issue: Memory error with large datasets
Solutions:

Reduce batch size: Process data in chunks
Use sparse matrices: For one-hot encoded data
Increase RAM: Or use cloud computing (Google Colab)
Issue: Model predictions always same class
Cause: Imbalanced data or improper scaling
Solution: Ensure SMOTE is applied and features are scaled

# 📦 Dependencies Explained

Core Libraries

- pandas
Purpose: Data manipulation and analysis
Used for: Loading CSV, data exploration, preprocessing
Why: Industry standard, powerful DataFrame operations
- numpy
Purpose: Numerical computing
Used for: Array operations, mathematical functions
Why: Fast, efficient, foundation for other libraries
- scikit-learn
Purpose: Machine learning algorithms
Used for: Models, metrics, preprocessing, train-test split
Why: Comprehensive, well-documented, production-ready
- imbalanced-learn
Purpose: Handling imbalanced datasets
Used for: SMOTE oversampling
Why: Specialized for class imbalance problems
- XGBoost
Purpose: Gradient boosting framework
Used for: High-performance classification
Why: State-of-the-art accuracy, fast training, production-ready
- seaborn
Purpose: Statistical data visualization
Used for: Bar plots, count plots
Why: Beautiful defaults, built on matplotlib
- joblib
Purpose: Model serialization
Used for: Saving/loading trained models
Why: Efficient for large numpy arrays, scikit-learn integration

requirements.txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
xgboost>=1.5.0
seaborn>=0.11.0
matplotlib>=3.4.0
joblib>=1.1.0

# 🤝 Contributing
Contributions are welcome! Here's how you can help:

Reporting Bugs
Check if the issue already exists
Provide detailed description with steps to reproduce
Include system information (OS, Python version)
Share error messages and stack traces
Suggesting Features
Open an issue with [Feature Request] tag
Describe the feature and its benefits
Provide examples or mockups if possible
Ideas for Enhancement:

Web interface with Flask/Streamlit
Real-time prediction API
Feature importance visualization
Hyperparameter tuning automation
Deep learning models (Neural Networks)
Explainable AI (SHAP values)
Pull Requests
Fork the repository
Create a feature branch: git checkout -b feature/AmazingFeature
Commit changes: git commit -m 'Add AmazingFeature'
Push to branch: git push origin feature/AmazingFeature
Open a Pull Request
Code Standards:

Follow PEP 8 style guide
Add docstrings to functions
Include unit tests for new features
Update README with new functionality

# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

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

# 👨‍💻 Author
Hasnain Haider

Email: hhnain1006@gmail.com
GitHub: @Hasnain006-nain
LinkedIn: Connect with me

# 🙏 Acknowledgments
scikit-learn - Comprehensive machine learning library
XGBoost - High-performance gradient boosting
imbalanced-learn - SMOTE and resampling techniques
Kaggle - Dataset inspiration and community
Stack Overflow - Problem-solving community
# 📞 Support
For issues, questions, or suggestions:

GitHub Issues: Open an issue
Email: hhnain1006@gmail.com
Discussions: Use GitHub Discussions for general questions

# 🚀 Future Enhancements
[ ] Web dashboard with Streamlit
[ ] REST API for predictions
[ ] Feature importance analysis
[ ] SHAP values for explainability
[ ] Automated hyperparameter tuning
[ ] Deep learning models (LSTM, Neural Networks)
[ ] Real-time data pipeline
[ ] Docker containerization
[ ] Cloud deployment (AWS/Azure/GCP)

⭐ If you found this project helpful, please give it a star!
Made with ❤️ by Hasnain Haider




















