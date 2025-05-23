"""# Decision Tree and Random Forest

Load the IRIS dataset. The dataset consists of 150 samples of iris flowers, each belonging to
one of three species (setosa, versicolor, or virginica). Each sample includes four features: sepal
length, sepal width, petal length, and petal width.

"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the IRIS dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

"""# Visualize the distribution of each feature and the class distribution.

"""

# Visualize the distribution of each feature
sns.pairplot(iris_df, hue='species')
plt.show()

# Visualize the class distribution
sns.countplot(x='species', data=iris_df)
plt.show()

"""# Encode the categorical target variable (species) into numerical values.

"""

from sklearn.preprocessing import LabelEncoder
# Encode the target variable using label encoder
label_encoder = LabelEncoder()
iris_df['species'] = label_encoder.fit_transform(iris_df['species'])

"""# Split the dataset into training and testing sets (use an appropriate ratio).

"""

from sklearn.model_selection import train_test_split
# Split the dataset
X = iris_df.drop(columns=['species'])
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""# Decision Tree Model

# Build a decision tree classifier using the training set.

"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Build the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

"""# Visualize the resulting decision tree.

"""

plt.figure(figsize=(20,10))
tree.plot_tree(dt_classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

"""# Make predictions on the testing set and evaluate the model's performance using appropriate metrics (e.g., accuracy, confusion matrix).

"""

from sklearn.metrics import accuracy_score, confusion_matrix
# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print(f"Decision Tree Accuracy: {accuracy_dt}")
print("Confusion Matrix:")
print(conf_matrix_dt)

"""# Random Forest Model

# Build a random forest classifier using the training set
"""

from sklearn.ensemble import RandomForestClassifier

# Build the random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test)
# Evaluate the model

accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf}")
print("Confusion Matrix:")
print(conf_matrix_rf)

"""# Tune the hyperparameters (e.g., number of trees, maximum depth) if necessary.

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20],     # Maximum depth of the trees
}

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')  # Use 5-fold cross-validation
grid_search.fit(X_train, y_train)


# Print the best hyperparameters and the best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use the best model for predictions
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate the best model's performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Accuracy (with best hyperparameters): {accuracy_rf}")
print("Confusion Matrix:")
conf_matrix_rf

"""# Make predictions on the testing set and evaluate the model's performance using appropriate metrics and compare it with the decision tree model."""

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

# Assuming X_test, y_test, dt_classifier, and rf_classifier are defined from previous code

# Decision Tree Evaluation
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print("Decision Tree Performance:")
print(f"Accuracy: {accuracy_dt}")
print("Confusion Matrix:\n", conf_matrix_dt)

# Random Forest Evaluation
y_pred_rf = best_rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_rf}")
print("Confusion Matrix:\n", conf_matrix_rf)

# Comparison
print("\nModel Comparison:")
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

if accuracy_rf > accuracy_dt:
    print("Random Forest performs better than Decision Tree.")
elif accuracy_dt > accuracy_rf:
    print("Decision Tree performs better than Random Forest.")
else:
    print("Both models have the same accuracy.")
