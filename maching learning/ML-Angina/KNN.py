import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Read the CSV file, skipping the first row and first column
data_frame = pd.read_csv("C:\\Users\\Robert Jiang\\Desktop\\maching learning\\Angina.csv", skiprows=[0])

# Drop samples containing missing values
data_frame.dropna(inplace=True)

# Extract feature and label columns
X = data_frame.iloc[:, 1:-1]  # Modify based on the actual column index of the data
y = data_frame.iloc[:, -1]

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle missing values
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the range of hyperparameters to tune
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

# Create a KNN classifier
knn = KNeighborsClassifier()

# Use GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Output the best hyperparameters and corresponding model performance
print("Best Parameters:", grid_search.best_params_)
print("Best Training Accuracy:", grid_search.best_score_)

# Retrain the model with the best hyperparameters
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)

# Predictions on the training set
y_train_pred = best_knn.predict(X_train)

# Predictions on the testing set
y_test_pred = best_knn.predict(X_test)

# Calculate accuracy on training and testing sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Confusion matrices for training and testing sets
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)
print("Training Confusion Matrix:")
print(train_cm)
print("Testing Confusion Matrix:")
print(test_cm)

# Classification reports for training and testing sets
train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)
print("Training Classification Report:")
print(train_report)
print("Testing Classification Report:")
print(test_report)
