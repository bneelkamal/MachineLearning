

# 🚀 Q2: Linear and Logistic Regression Adventure

Welcome to **Q2: Linear and Logistic Regression**! This project dives into the world of regression analysis, tackling both linear and logistic flavors. Using datasets `linear_regression_dataset.csv` and `logistic_regression_dataset.csv`, we predict relationships and outcomes with a mix of built-in tools and custom code. Ready to explore? Let’s roll! 🌟

![Regression Banner](https://raw.githubusercontent.com/bneelkamal/MachineLearning/main/images/q2_banner.png)  
*Predicting the future, one line at a time!*

---

## 🎯 Project Overview

This project is split into two exciting tasks:
- **2a: Linear Regression** (6 Marks): Predict weight from height using both sklearn’s `LinearRegression` and a manual implementation.
- **2b: Logistic Regression** (4 Marks): Predict purchase likelihood based on gender, age, and salary with sklearn’s `LogisticRegression`.

---

## 🛠️ Tasks & Implementation

### 2a: Linear Regression Task
**Dataset**: `linear_regression_dataset.csv` (Height vs. Weight)  
**Objective**: Predict weight (dependent variable) from height (independent variable).

#### Inbuilt Linear Regression (sklearn)
- **Steps**:
  1. Loaded the dataset and split it into 80% training and 20% testing sets.
  2. Trained a `LinearRegression` model from `sklearn.linear_model`.
  3. Extracted coefficients: **Intercept** and **Slope**.
  4. Plotted a scatter of height vs. weight with the regression line.
- **Coefficients**: Printed for insight into the model’s fit.

![Linear Regression Sklearn](h[ttps://raw.githubusercontent.com/bneelkamal/MachineLearning/main/images/linear_sklearn.png](https://github.com/bneelkamal/MachineLearning/blob/main/Projects/images/linear_regression_manual.png))  
*Sklearn’s take on the height-weight relationship!*

#### Manual Linear Regression
- **Steps**:
  1. Built a custom function using the normal equation: `(X^T X)^(-1) X^T y`.
  2. Calculated intercept and slope from the training data.
  3. Made predictions on the test set.
  4. Plotted the scatter with the manual regression line.
- **Result**: Nearly identical to sklearn’s output, proving the math holds up!

![Linear Regression Manual](https://raw.githubusercontent.com/bneelkamal/MachineLearning/main/images/linear_manual.png)  
*Handcrafted precision in action!*

#### Comparison
- **Graphical**: Overlay of both regression lines shows they’re almost indistinguishable.
- **Coefficients**: Very close values between sklearn and manual methods—consistency confirmed!

![Comparison Plot](https://raw.githubusercontent.com/bneelkamal/MachineLearning/main/images/linear_comparison.png)  
*Spot the difference? Neither can we!*

---

### 2b: Logistic Regression Task
**Dataset**: `logistic_regression_dataset.csv` (User ID, Gender, Age, Estimated Salary, Purchased)  
**Objective**: Predict if a user will purchase a product based on Gender, Age, and Salary.

#### Data Preprocessing
- Dropped `User ID` (irrelevant for prediction).
- Converted `Gender` to binary: Male=1, Female=0.
- Defined features (`Gender`, `Age`, `Estimated Salary`) and target (`Purchased`).
- Split data: 80% training, 20% testing.

#### Model Training
- Used `LogisticRegression` from `sklearn.linear_model`.
- Trained on the training set, predicted on the test set.

#### Evaluation
- **Confusion Matrix**: Counts of TP, TN, FP, FN—visualized as a heatmap.
- **Accuracy**: Achieved ~88.75%—pretty solid!
- **Classification Report**: Precision, recall, and F1-score for both classes (0 and 1).

![Confusion Matrix](https://raw.githubusercontent.com/bneelkamal/MachineLearning/main/images/logistic_confusion.png)  
*Breaking down the predictions!*

---

## 📊 Results
- **Linear Regression**: Both implementations nailed the height-weight relationship with near-identical lines and coefficients.
- **Logistic Regression**: Predicted purchases with ~88.75% accuracy, backed by a detailed confusion matrix and classification metrics.

---

## 🧰 Tools & Libraries
- **Python**: The core engine.
- **Pandas**: Data loading and prep.
- **NumPy**: Math magic for manual regression.
- **Scikit-learn**: Inbuilt models and metrics.
- **Matplotlib/Seaborn**: Plotting the story.

![Python Badge](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Pandas Badge](https://img.shields.io/badge/Pandas-1.5+-orange.svg) ![Scikit-learn Badge](https://img.shields.io/badge/Scikit--learn-1.3+-green.svg)

---

## 🚀 How to Run
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/bneelkamal/MachineLearning.git
