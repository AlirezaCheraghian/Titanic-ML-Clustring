# Titanic Passenger Survival Analysis with Clustering

## Overview
This project analyzes Titanic passenger data using clustering techniques (HDBSCAN) and evaluates their impact on survival prediction models. The workflow includes:
1. Data preprocessing and cleaning
2. Feature engineering using HDBSCAN clustering
3. Survival prediction using Logistic Regression and KNN models
4. Performance evaluation with RMSE and timing metrics

## Key Features
- **Data Preprocessing**:
  - Handled missing values (Age filled with mean, Embarked rows dropped)
  - Encoded categorical features (Sex, Embarked)
  - Scaled numerical features (Age, Fare)
- **Clustering**:
  - Applied HDBSCAN to create cluster features for Age and Fare
  - Identified 38 unique age clusters and 40 fare clusters
- **Modeling**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN) with k=1
- **Evaluation**:
  - Root Mean Squared Error (RMSE)
  - Training and inference timing metrics

## Results
| Model               | RMSE     | Training Time | Test Time |
|---------------------|----------|---------------|-----------|
| Logistic Regression | 0.3966   | 0.3396 sec    | 0.0098 sec|
| KNN (k=1)           | 0.5139   | 0.0045 sec    | 0.0045 sec|

## Dependencies
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - hdbscan
  - matplotlib

## File Structure
titanic/
├── titanic.ipynb # Main Jupyter notebook with analysis code
├── titanic.csv # Dataset file
└── README.md # Project documentation
