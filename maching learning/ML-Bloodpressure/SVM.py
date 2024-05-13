import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read CSV data
data = pd.read_csv("C:\\Users\\Robert Jiang\\Desktop\\maching learning\\Bloodpressure.csv")

# Drop samples containing missing values
data.dropna(inplace=True)

# Remove the first row and first column
data = data.iloc[1:, 1:]

# Split the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle missing data
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Class weights
class_weights = dict()
unique_classes, class_counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
for i in range(len(unique_classes)):
    class_weights[unique_classes[i]] = total_samples / (len(unique_classes) * class_counts[i])

# Create SVM model
model = SVC(class_weight=class_weights)

# Evaluate model performance using cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Accuracy:", np.mean(cv_scores))

# Train the model using the entire training set
model.fit(X_train, y_train)

# Predictions on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy on training and testing sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print accuracy on training and testing sets
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Calculate confusion matrices for testing and training sets
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Print confusion matrices for testing and training sets
print("Training Confusion Matrix:")
print(train_cm)
print("Testing Confusion Matrix:")
print(test_cm)

# Calculate classification reports for testing and training sets
train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)

# Print classification reports for testing and training sets
print("Training Classification Report:")
print(train_report)
print("Testing Classification Report:")
print(test_report)
