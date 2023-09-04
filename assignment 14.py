# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 09:51:11 2023

@author: India
"""
# Problem Statement 1: A cloth manufacturing company
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Company_Data.csv')

# Define a threshold for categorizing 'Sales' into high/low categories
threshold = 8.0
data['Sales_Category'] = data['Sales'].apply(lambda x: 'High' if x >= threshold else 'Low')

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)

# Select independent variables and target variable
X = data_encoded.drop(['Sales', 'Sales_Category'], axis=1)
y = data_encoded['Sales_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# You can also print other evaluation metrics if needed
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

### data visualisation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
# Pairplot: Visualize relationships between numeric variables
sns.pairplot(data, hue='Sales_Category', diag_kind='kde')
plt.show()

# Correlation Heatmap: Visualize the correlation between numeric variables
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Boxplots: Visualize the distribution of numeric variables by Sales Category
plt.figure(figsize=(12, 6))
sns.boxplot(x='Sales_Category', y='Income', data=data)
plt.title('Income Distribution by Sales Category')
plt.show()

# Countplot: Visualize the count of categorical variables by Sales Category
plt.figure(figsize=(8, 6))
sns.countplot(x='ShelveLoc', hue='Sales_Category', data=data)
plt.title('Shelve Location Count by Sales Category')
plt.show()

# Barplot: Visualize the mean 'Advertising' budget by Sales Category
plt.figure(figsize=(8, 6))
sns.barplot(x='Sales_Category', y='Advertising', data=data, ci=None)
plt.title('Mean Advertising Budget by Sales Category')
plt.show()

# Histogram: Visualize the distribution of 'Age' by Sales Category
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Age', hue='Sales_Category', element='step', common_norm=False, kde=True)
plt.title('Age Distribution by Sales Category')
plt.show()







#####  Problem statement 2:  model on fraud data 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Fraud_check.csv')

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Undergrad', 'Marital.Status', 'Urban'], drop_first=True)

# Create the binary target variable 'Taxable_Income_Category'
threshold = 30000
data_encoded['Taxable_Income_Category'] = data_encoded['Taxable.Income'].apply(lambda x: 'Risky' if x <= threshold else 'Good')

# Select independent variables and target variable
X = data_encoded.drop(['Taxable.Income', 'Taxable_Income_Category'], axis=1)
y = data_encoded['Taxable_Income_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# You can also print other evaluation metrics if needed
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



### data visualisation
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 

# Countplot for 'Undergrad'
plt.figure(figsize=(8, 6))
sns.countplot(x='Undergrad', data=data, hue='Taxable.Income')
plt.title('Count of Undergraduates by Taxable Income')
plt.xlabel('Undergrad')
plt.ylabel('Count')
plt.show()

# Countplot for 'Marital.Status'
plt.figure(figsize=(10, 6))
sns.countplot(x='Marital.Status', data=data, hue='Taxable.Income')
plt.title('Count of Marital Status by Taxable Income')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()




