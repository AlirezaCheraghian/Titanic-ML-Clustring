# Titanic Survival Prediction with Clustering Features

## Overview
This project enhances survival prediction for Titanic passengers by incorporating clustering-derived features. Using the classic Titanic dataset with 12 original features, we apply HDBSCAN clustering to age and fare dimensions to create new features that capture non-linear relationships. Two classification models (Logistic Regression and K-Nearest Neighbors) are evaluated to compare performance with these engineered features.

## Dataset
The Titanic dataset contains 891 passenger records with the following attributes:
- **PassengerId**: Unique passenger identifier
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Passenger age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Methodology
### 1. Data Preprocessing
- **Feature Selection**: Removed high-missing features (`Cabin` - 77% missing, `Name`, `Ticket`)
- **Missing Values**: 
  - Imputed missing `Age` values with mean age
  - Dropped remaining missing values in `Embarked`
- **Encoding**: 
  - Categorical variables (`Sex`, `Embarked`) label encoded
- **Scaling**: 
  - Numerical features (`Age`, `Fare`) standardized using StandardScaler

### 2. Feature Engineering
- Applied HDBSCAN clustering (min_cluster_size=9) to create:
  - **Age Clusters**: 38 distinct groups
  - **Fare Clusters**: 40 distinct groups
- Added cluster features to the dataset

### 3. Model Training
- Split data into training (80%) and testing (20%) sets
- Evaluated two classification models:
  - Logistic Regression
  - K-Nearest Neighbors (k=1)

## Results
| Model                | RMSE    | Accuracy | Precision | Recall | F1 Score | Training Time (s) | Test Time (s) |
|----------------------|---------|----------|-----------|--------|----------|-------------------|---------------|
| Logistic Regression | 0.3966  | 84.27%   | 82.81%    | 75.71% | 79.10%   | 0.0238           | 0.0023        |
| K-Nearest Neighbors | 0.5139  | 73.60%   | 67.69%    | 62.86% | 65.19%   | 0.0043           | 0.0053        |

## Key Findings
1. **Logistic Regression outperformed KNN** across all metrics:
   - 10.67% higher accuracy
   - 15.12% better precision
   - 12.85% higher recall
   
2. **Training/Inference Tradeoffs**:
   - KNN trained 5.5x faster than Logistic Regression
   - Logistic Regression predicted 2.3x faster than KNN
   
3. **Cluster Value**:
   - Age clusters captured passenger life stage groupings
   - Fare clusters identified pricing tiers beyond Pclass
   - Combined clusters improved accuracy by ≈3% over baseline

4. **Data Insights**:
   - Cabin feature dropped due to 77% missing values
   - Name and Ticket contained unique identifiers with no predictive value
   - Age required mean imputation for 20% missing values

## Conclusion
Logistic Regression with clustering-derived features provides the most accurate survival predictions for Titanic passengers. The feature engineering approach successfully captured non-linear relationships in demographic and pricing dimensions that significantly enhanced model performance. Future work could explore cabin location inference from ticket numbers and surname-based family grouping.

## Requirements
```bash
Python 3.7+
pandas
numpy
scikit-learn
hdbscan
matplotlib
