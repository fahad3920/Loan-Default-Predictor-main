# **Credit Risk Prediction** 
## Overview
This project focuses on predicting credit risk by analyzing loan applicant data. The notebook follows a comprehensive workflow from data loading and cleaning to exploratory data analysis (EDA), feature engineering, and machine learning model implementation. The goal is to build a model that can accurately predict whether a loan application is likely to default on a loan.

## Dataset
The dataset used is train_u6lujuX_CVtuZ9i (1).csv, containing 614 entries with 13 features:

- Loan_ID: Unique identifier for each loan application

- Gender: Applicant's gender (Male/Female)

- Married: Marital status (Yes/No)

- Dependents: Number of dependents (0, 1, 2, 3+)

- Education: Education level (Graduate/Not Graduate)

- Self_Employed: Self-employment status (Yes/No)

- ApplicantIncome: Applicant's income

- CoapplicantIncome: Co-applicant's income

- LoanAmount: Loan amount in thousands

- Loan_Amount_Term: Term of loan in days

- Credit_History: Credit history meets guidelines (1.0) or not (0.0)

- Property_Area: Location of property (Urban/Rural/Semiurban)

- Loan_Status: Target variable (Y/N)

## Data Cleaning and Preprocessing
### Handling Missing Values:

#### Filled missing categorical values with mode:
#### Filled missing numerical values:
#### Dropped rows with missing Credit_History (50 rows)

## Feature Engineering:

Created Loan_Status_Numeric (1 for 'Y', 0 for 'N') for correlation analysis
### Correlation Analysis:
Found Credit_History has the highest correlation (0.56) with Loan_Status

## Exploratory Data Analysis (EDA)
### Key visualizations and insights:

#### Gender Distribution: 82% applicants are male, 18% female

#### Loan approval rates: Male: 69.1% ,Female: 64.4%

#### Marital Status: 63% applicants are married

#### Loan approval rates: Married: 71.2% ,Single: 62.8%

#### And Much More

## Machine Learning Implementation

Handling Class Imbalance:

Applied SMOTE (oversampling) and RandomUnderSampler

Compared results with and without balancing

### Models Implemented
Baseline Models:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Support Vector Machine (SVM)

### Ensemble Models:

Random Forest

Extra Trees

AdaBoost

Gradient Boosting

XGBoost

### Model Evaluation:

### Metrics:

Accuracy

Precision

Recall

F1-score

Confusion matrices

Hyperparameter Tuning:

GridSearchCV for optimal parameters

Focused on key parameters for each model

### Key Findings
Best performing model: [Votting Classifier] with [83]%

## Technical Stack  
- **Python Libraries**: Pandas, Scikit-learn, XGBoost, Imbalanced-learn  
- **Visualization**: Plotly, Seaborn , Matplolib
- **Deployment**: Streamlit web app (`app.py` included!)  
