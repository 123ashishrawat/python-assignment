# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:27:26 2023

@author: India
"""
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('bank-full.csv')


label_encoder = LabelEncoder() # Preprocess the data ,  # Encode categorical variables using Label Encoding
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Split the data into training and testing sets
X = data.drop(columns=['y'])  # Features
y = data['y']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)



############ data visualisation

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the target variable 'y'
plt.figure(figsize=(6, 4))
sns.countplot(data['y'])
plt.title('Distribution of Subscribed Term Deposit (Target Variable)')
plt.xlabel('Subscribed Term Deposit')
plt.ylabel('Count')
plt.show()

# Plot a correlation matrix of numerical features
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
correlation_matrix = data[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Plot a bar chart for the 'job' column
plt.figure(figsize=(10, 6))
sns.countplot(data['job'], hue=data['y'])
plt.title('Subscription by Job')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Subscribed Term Deposit', loc='upper right', labels=['No', 'Yes'])
plt.show()

# Plot a pairplot of numerical features
sns.pairplot(data, hue='y', vars=numerical_features)
plt.suptitle('Pairplot of Numerical Features by Subscription', y=1.02)
plt.show()




















