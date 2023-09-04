# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 09:51:09 2023

@author: India
"""
### Problem Statement 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('glass.csv')

# Select features and target variable
X = data.iloc[:,0:9]

y = LabelEncoder().fit_transform(data['Type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the KNN model
k = 3  # for k=3 we are getting higher accuracy score
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# You can also print other evaluation metrics if needed
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))









# Problem Statement 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Zoo.csv')

# Select features and target variable
X = data.iloc[:,1:17]

y = data['type']  # Assuming 'type' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the KNN model
k = 5  # You can experiment with different values of K
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# You can also print other evaluation metrics if needed
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))











