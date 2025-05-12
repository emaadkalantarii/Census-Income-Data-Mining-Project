# Census Income Prediction Project

## Table of Contents
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Methodology](#methodology)
    * [Data Loading and Exploration](#data-loading-and-exploration)
    * [Data Cleaning](#data-cleaning)
    * [Data Preprocessing](#data-preprocessing)
    * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    * [Model Training and Evaluation](#model-training-and-evaluation)
* [Results](#results)
* [Visualizations](#visualizations)
* [Libraries Used](#libraries-used)
* [How to Run](#how-to-run)

## Introduction
This project is a part of a Knowledge Discovery and Data Mining course. The primary goal is to predict whether an individual's income exceeds $50K/yr based on census data. This is a binary classification task utilizing various demographic and socio-economic features.

## Dataset
The project uses the "Census Income" dataset, also known as the "Adult" dataset, from the UCI Machine Learning Repository.
- **Source:** [UCI Machine Learning Repository: Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income)
- **Abstract:** Predict whether income exceeds $50K/yr based on census data.
- **Number of Instances:** 48,842 (before cleaning)
- **Number of Features:** 14 (plus a target variable 'income')
- **Feature Types:** Categorical and Integer.
- **Target Variable:** `income` (binary: <=50K or >50K)
- **Missing Values:** Yes, present in 'workclass', 'occupation', and 'native-country'.

## Methodology

### Data Loading and Exploration
- The dataset was fetched using the `ucimlrepo` Python library.
- Initial exploration involved examining the metadata and variable information provided by the repository.

### Data Cleaning
- **Target Variable (`income`):** The target variable had inconsistent formatting (e.g., `<=50K.` and `>50K.`). These were cleaned by stripping leading/trailing whitespace and removing the trailing period to have consistent labels (`<=50K`, `>50K`).
- **Missing Values:** Rows containing missing values (NaN) were identified in 'workclass', 'occupation', and 'native-country'. These rows were dropped from the dataset to ensure data quality for model training. After dropping missing values, the dataset contained 47,621 instances.

### Data Preprocessing
1.  **Feature Separation:** Features were identified and separated into numerical (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and categorical (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`) types.
2.  **Numerical Feature Scaling:** Numerical features were scaled using `MinMaxScaler` from `sklearn.preprocessing` to normalize their ranges between 0 and 1. This helps in improving the performance of certain machine learning algorithms.
3.  **Categorical Feature Encoding:** Categorical features were converted into numerical format using one-hot encoding via `pd.get_dummies`, with `drop_first=True` to avoid multicollinearity.
4.  **Target Variable Encoding:** The cleaned binary target variable (`y`) was encoded into numerical values (0 and 1) using `LabelEncoder` from `sklearn.preprocessing`.

### Exploratory Data Analysis (EDA)
- Descriptive statistics for numerical features were generated to understand their distributions (mean, std, min, max, quartiles).
- A stacked bar chart was created to visualize the relationship between 'sex' and 'income'.
- A correlation matrix heatmap for numerical features was generated to understand linear relationships between them.
- The distribution of the 'age' feature was visualized using a histogram with a Kernel Density Estimate (KDE).

### Model Training and Evaluation
The preprocessed dataset was split into training (70%) and testing (30%) sets using `train_test_split` with `random_state=42` for reproducibility.

Three different classification models were trained and evaluated:
1.  **Logistic Regression:**
    * The effect of the regularization parameter `C` on accuracy was explored by training models with `C` values of \[0.01, 0.1, 1, 10, 100].
    * The model with `C=100` was selected for final evaluation.
2.  **Random Forest Classifier:**
    * The impact of `n_estimators` (number of trees) and `max_depth` on accuracy was investigated.
    * The model with `n_estimators=200` and `max_depth=20` was selected for final evaluation.
3.  **XGBoost Classifier:**
    * The influence of `learning_rate` and `n_estimators` on accuracy was analyzed.
    * The model with `learning_rate=0.30` and `n_estimators=100` was selected for final evaluation.

**Evaluation Metrics:**
For each model, the following metrics were calculated on the test set:
- Accuracy
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R2 Score
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1 Score (macro-averaged)
- Confusion Matrix

## Results
The performance of the models on the test set was as follows:
- **Logistic Regression (C=100):**
    - Accuracy: ~84.79%
    - F1 Score (macro): ~0.780
- **Random Forest (n_estimators=200, max_depth=20):**
    - Accuracy: ~85.98%
    - F1 Score (macro): ~0.793
- **XGBoost (learning_rate=0.3, n_estimators=100):**
    - Accuracy: ~86.95%
    - F1 Score (macro): ~0.814

XGBoost Classifier demonstrated the highest accuracy and F1-score among the three models.

## Visualizations
The project includes several visualizations to aid in understanding the data and model performance:
- **Stacked Bar Chart:** Sex vs. Income.
- **Correlation Matrix Heatmap:** For numerical features.
- **Histogram:** Age distribution.
- **Accuracy Plots:**
    - Logistic Regression: Accuracy vs. Regularization Parameter (C).
    - Random Forest: Accuracy vs. Number of Estimators and Accuracy vs. Maximum Depth.
    - XGBoost: Accuracy vs. Learning Rate and Accuracy vs. Number of Trees.
- **Confusion Matrices:** For each of the final models (Logistic Regression, Random Forest, XGBoost) to show true positives, true negatives, false positives, and false negatives.

## Libraries Used
- `ucimlrepo`: For fetching the dataset.
- `pandas`: For data manipulation and analysis.
- `sklearn` (scikit-learn): For data preprocessing (MinMaxScaler, OneHotEncoder, LabelEncoder), model training (LogisticRegression, RandomForestClassifier), model splitting (train_test_split), and evaluation metrics (accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay).
- `matplotlib.pyplot`: For plotting visualizations.
- `seaborn`: For enhanced visualizations (heatmap).
- `xgboost`: For the XGBoost Classifier model.
- `math` (specifically `sqrt`): For calculating RMSE.

## How to Run
1.  Ensure you have Python installed.
2.  Install the necessary libraries:
    ```bash
    pip install ucimlrepo pandas scikit-learn matplotlib seaborn xgboost
    ```
3.  Download the `Census_Income_Data_Mining_Project.ipynb` file.
4.  Open and run the Jupyter Notebook in an environment like Jupyter Lab, Jupyter Notebook, or Google Colab. The first cell installs `ucimlrepo` if it's not already present.
