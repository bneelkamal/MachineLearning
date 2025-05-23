"""#SVM
Use the dataset “Bank_Personal_Loan_Modelling.csv”

#Store the dataset in your google drive and in Colab file load the dataset from your drive.

"""

#mount google drive to read the files from drive
from google.colab import drive
import pandas as pd #For data processing
drive.mount('/content/drive', force_remount=True)
bank_df = pd.read_csv("/content/drive/MyDrive/Bank_Personal_Loan_Modelling.csv")

# Commented out IPython magic to ensure Python compatibility.
#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score # Performance measure – Accuracy
from sklearn import preprocessing

"""#Check the shape and head of the dataset."""

print(bank_df.shape)
bank_df.head()

"""#Age, Experience, Income, CCAvg, Mortgage, Securities are the features and Credit Card is your Target Variable.

Target Variable = CreditCard

Features = (Age, Experience, Income, CCAvg, Mortgage, Securities )

#3.3.i. Take any 3 features from the six features given above

Lets take the feature variables: Income, CCAvg, Mortgage

X = (Income, CCAvg, Mortgage )

#Store features and targets into a separate variable
"""

x = bank_df[['Income', 'CCAvg', 'Mortgage']]
y = bank_df['CreditCard']

"""#Look for missing values in the data, if any, and address them accordingly."""

#Check for missing values
bank_df.isnull().values.any()
#no missing values found
bank_df.describe().T

"""#3.3.iv iv. Plot a 3D scatter plot using Matplotlib.

"""

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(bank_df['Income'], bank_df['CCAvg'], bank_df['Mortgage'],c=bank_df['CreditCard'], cmap='viridis', marker='o')
ax.set_xlabel('Income')
ax.set_ylabel('CCAvg')
ax.set_zlabel('Mortgage')
plt.title("3D Scatter Plot: Income,CCAvg, Mortgage (Color ~ Credit Card)")
plt.colorbar(sc, label="Credit Card")
plt.show()

"""#Split the dataset into 80:20. (3 features and 1 target variable).

"""

# Split the dataset into 80:20 (3 features and 1 target variable)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

"""#Train the model using scikit learn SVM API (LinearSVC) by setting the regularization parameter C as C = {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}.

#For each value of C Print the score on test data
"""

from sklearn.svm import LinearSVC

C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

print("Linear SVC Kernal:")
for C in C_values:
    # Create and train the LinearSVC model
    svm_model = LinearSVC(C=C, random_state=42 , class_weight ='balanced')  # Set random_state for reproducibility
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"For C = {C}, Accuracy: {accuracy:.4f}")

#using different kernals - Linear SVC
from sklearn.svm import SVC
print("RBF Kernal:")
for C in C_values:
# Create and train the SVC model with an RBF kernel
  rbf_svm_model = SVC(kernel='rbf', C=C, random_state=42, class_weight='balanced')
  rbf_svm_model.fit(X_train, y_train)
# Make predictions on the test set
  rbf_y_pred = rbf_svm_model.predict(X_test)
# Evaluate the model
  rbf_accuracy = accuracy_score(y_test, y_pred)
  print(f"For C = {C}, Accuracy: {rbf_accuracy:.4f}")


# Example with a polynomial kernel
# degree specifies the degree of the polynomial
print("Poly Kernal:")
for C in C_values:
# Create and train the SVC model with an Poly kernel
  poly_svm_model = SVC(kernel='poly', C=C, random_state=42, class_weight='balanced', degree=3)
  poly_svm_model.fit(X_train, y_train)
# Make predictions on the test set
  poly_y_pred = rbf_svm_model.predict(X_test)
# Evaluate the model
  poly_accuracy = accuracy_score(y_test, y_pred)
  print(f"For C = {C}, Accuracy: {poly_accuracy:.4f}")

"""#3.5.ii. Make the prediction on test data

"""

#Make predictions on the test set using the best model based on accuracy - Linear SVC , Best C Value = 0.001
best_C = 0.001
svm_model = LinearSVC(C=best_C, random_state=42 , class_weight ='balanced')
svm_model.fit(X_train,y_train)
y_pred = svm_model.predict(X_test)
#Print the predictions
print(y_pred)

"""#Print confusion matrix and classification report

"""

from sklearn.metrics import confusion_matrix, classification_report

# Calculate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate and print the classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
report

"""#Use gridSearchCV a cross-validation technique to find the best regularization parameters (i.e.: the best value of C).

"""

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for C
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Create a LinearSVC model
svm_model = LinearSVC(random_state=42, class_weight= 'balanced',)

# Create GridSearchCV object
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')  # Use 5-fold cross-validation

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy with best C: {accuracy:.4f}")

# Print confusion matrix and classification report for the best model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
