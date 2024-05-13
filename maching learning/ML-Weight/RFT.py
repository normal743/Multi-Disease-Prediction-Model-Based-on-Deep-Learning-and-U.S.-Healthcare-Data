import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Read the CSV file, skipping the first row
data_frame = pd.read_csv("C:\\Users\\Robert Jiang\\Desktop\\maching learning\\Weight.csv", skiprows=[0])

# Drop samples containing missing values
data_frame.dropna(inplace=True)

# Extract feature and label columns
X = data_frame.iloc[:, :-1]
y = data_frame.iloc[:, -1]

# Handle missing values
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [10, 30, 50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5]
}

# Create a random forest classifier
rf = RandomForestClassifier()

# Create a grid search object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# Perform grid search on the training set
grid_search.fit(X_train, y_train)

# Print the best parameter combination
print("Best Parameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the training set
y_train_pred = best_model.predict(X_train)

# Make predictions on the testing set
y_test_pred = best_model.predict(X_test)

# Compute accuracy on training and testing sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Compute confusion matrices for training and testing sets
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)
print("Training Confusion Matrix:")
print(train_cm)
print("Testing Confusion Matrix:")
print(test_cm)

# Compute classification reports for training and testing sets
train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)
print("Training Classification Report:")
print(train_report)
print("Testing Classification Report:")
print(test_report)
