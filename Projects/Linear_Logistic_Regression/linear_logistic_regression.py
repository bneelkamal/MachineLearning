# Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
lr_url = "https://raw.githubusercontent.com/bneelkamal/MachineLearning/refs/heads/main/DataFiles/linear_regression_dataset.csv"
df = pd.read_csv(lr_url)
X = df.iloc[:, 0].values.reshape(-1, 1)  
Y = df.iloc[:, 1].values                 

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 1. Using sklearn's LinearRegression
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, Y_train)
Y_pred_sklearn = sklearn_model.predict(X_test)

# Plot results
plt.scatter(X_test, Y_test, color='blue', label='Test Data')
plt.plot(X_test, Y_pred_sklearn, color='red', linewidth=1, label='Sklearn Model')
plt.xlabel("Height - Independent Variable")
plt.ylabel("Weight - Dependent Variable")
plt.title("Linear Regression using inbuilt function in sklearn")
plt.legend()
plt.show()

# Print coefficients
print(f"Sklearn Coefficients: Intercept = {sklearn_model.intercept_:.2f}, Slope = {sklearn_model.coef_[0]:.2f}")

# Manual implementation
def manual_linear_regression(X, Y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    return beta

beta = manual_linear_regression(X_train, Y_train)
Y_pred_manual = beta[0] + beta[1] * X_test

# Print the plot
plt.plot(X_test, Y_pred_manual, color='green', linestyle='dashed', linewidth=2, label='Manual Model')
plt.xlabel("Height - Independent Variable")
plt.ylabel("Weight - Dependent Variable")
plt.title("Linear Regression using Manual Implementation")
plt.legend()
plt.show()
print(f"Manual Coefficients: Intercept = {beta[0]:.2f}, Slope = {beta[1]:.2f}")

# Compare results graphically
plt.scatter(X_test, Y_test, color='blue', label='Test Data')
plt.plot(X_test, Y_pred_sklearn, color='red', linewidth=2, label='Sklearn Model')
plt.plot(X_test, Y_pred_manual, color='green', linestyle='dashed', linewidth=2, label='Manual Model')
plt.xlabel("Height - Independent Variable")
plt.ylabel("Weight - Dependent Variable")
plt.title("Linear Regression: Sklearn vs. Manual")
plt.legend()
plt.show()


# Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
lr_url = "https://raw.githubusercontent.com/bneelkamal/MachineLearning/refs/heads/main/DataFiles/logistic_regression_dataset.csv"
df = pd.read_csv(lr_url)

# Data Preprocessing
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == "Male" else 0)
df = df.drop("User ID", axis=1)
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the logistic regression classifier
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Make predictions
y_pred = log_reg_model.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:")
print(report)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
