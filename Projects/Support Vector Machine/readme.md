---
# Bank Personal Loan Modeling with SVM

This project explores the "Bank\_Personal\_Loan\_Modelling.csv" dataset to predict customer credit card adoption using Support Vector Machine (SVM) models. We'll walk through data loading, exploration, feature selection, and model training with different SVM kernels and hyperparameter tuning.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Running the Code](#running-the-code)
4.  [Methodology](#methodology)
    * [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    * [Dataset Exploration](#dataset-exploration)
    * [Feature and Target Selection](#feature-and-target-selection)
    * [Missing Value Handling](#missing-value-handling)
    * [3D Scatter Plot Visualization](#3d-scatter-plot-visualization)
    * [Data Splitting](#data-splitting)
    * [Model Training and Evaluation with LinearSVC](#model-training-and-evaluation-with-linearsvc)
    * [Hyperparameter Tuning with GridSearchCV](#hyperparameter-tuning-with-gridsearchcv)
5.  [Results and Conclusion](#results-and-conclusion)

---

## Project Overview

The goal of this project is to build and evaluate a classification model using **Support Vector Machines (SVM)** to predict whether a customer will accept a personal loan. We'll use the "Bank\_Personal\_Loan\_Modelling.csv" dataset and focus on `Income`, `CCAvg`, and `Mortgage` as features, with `CreditCard` as the target variable.

---

## Dataset

The dataset used in this project is "Bank\_Personal\_Loan\_Modelling.csv". It contains 5000 rows and 14 columns, providing various customer attributes.

---

## Getting Started

### Prerequisites

* **Google Colab:** This notebook is designed to be run in Google Colab, leveraging its integration with Google Drive.
* **Python 3.x**

### Installation

No specific installation steps are required beyond what's typically available in a Google Colab environment. The necessary libraries (pandas, numpy, seaborn, matplotlib, scikit-learn) are usually pre-installed.

### Running the Code

1.  **Upload to Google Drive:** Ensure the "Bank\_Personal\_Loan\_Modelling.csv" file is uploaded to your Google Drive. Update the path in the code if your file is not in `/content/drive/MyDrive/T1/ML/ASSIGNMENT1/Bank_Personal_Loan_Modelling.csv`.
2.  **Open in Colab:** Open the Python script (e.g., `bank_loan_prediction.ipynb` or the `.py` file containing the provided code) in Google Colab.
3.  **Run Cells:** Execute each cell sequentially to follow the analysis and model training steps.

---

## Methodology

### Data Loading and Preprocessing

The "Bank\_Personal\_Loan\_Modelling.csv" dataset is loaded from Google Drive into a pandas DataFrame. Essential libraries for data manipulation, analysis, and visualization are imported.

### Dataset Exploration

We examine the dataset's dimensions using `bank_df.shape` (5000 rows, 14 columns) and preview the first few rows with `bank_df.head()` to understand its structure.

### Feature and Target Selection

* **Features (Independent Variables):** `Income`, `CCAvg`, and `Mortgage`
* **Target (Dependent Variable):** `CreditCard`

### Missing Value Handling

A check for missing values using `bank_df.isnull().values.any()` confirmed that the dataset is clean with no missing data.

### 3D Scatter Plot Visualization

A 3D scatter plot is generated using Matplotlib to visually explore the relationship between `Income`, `CCAvg`, and `Mortgage`, with points colored based on the `CreditCard` adoption status. This helps in understanding the data distribution in a multi-dimensional space.

### Data Splitting

The dataset is split into training and testing sets with an **80:20 ratio** using `train_test_split`. This ensures that the model is trained on a significant portion of the data and evaluated on unseen data, providing a robust performance assessment.

### Model Training and Evaluation with LinearSVC

A **Linear SVM model (LinearSVC)** is trained, and its performance is evaluated with various regularization parameters (C): `[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]`.
For each `C` value, the model's accuracy on the test data is printed. **The best `C` value found to be `0.001` with Linear SVC.**

Additionally, **RBF** and **Polynomial** kernels are also tested, and a comparison of their accuracies is provided.
Predictions are made on the test data using the best Linear SVC model. The **confusion matrix** and **classification report** are generated to provide a comprehensive evaluation of the model's performance, including precision, recall, and F1-score.

### Hyperparameter Tuning with GridSearchCV

**GridSearchCV** is employed to systematically find the optimal regularization parameter (`C`) using **5-fold cross-validation**. The same `C` values are used in the parameter grid, with `accuracy` as the scoring metric.

* The **best `C` value identified by GridSearchCV is `0.0001`**.
* The model is re-trained with this optimal `C` value and its performance on the test data (accuracy, confusion matrix, and classification report) is printed, showcasing the final model's effectiveness.

---

## Results and Conclusion

* **Linear Kernel** provided decent accuracy but might be limited by linear decision boundaries.
* **RBF Kernel** generally achieved higher accuracy, capturing non-linear relationships.
* **Polynomial Kernel** showed comparable or slightly better accuracy than RBF in some cases, but was more computationally expensive.
* Higher `C` values generally improved training accuracy but risked overfitting.
* **GridSearchCV** proved crucial in identifying the optimal `C` for each kernel, effectively balancing bias and variance for better generalization.
* Regularization played a significant role in preventing overfitting and achieving optimal model performance.

This project demonstrates a complete workflow for building and evaluating SVM models for credit card adoption prediction, highlighting the importance of feature selection, data splitting, model selection, and hyperparameter tuning for achieving robust performance.
