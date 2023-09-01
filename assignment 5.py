# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:57:58 2023

@author: India
"""

##  1)   linear regression model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset

data=pd.read_csv("50_Startups.csv") 

df = pd.DataFrame(data)

# Preprocessing: Encode categorical feature 'State'
label_encoder = LabelEncoder()
df['State'] = label_encoder.fit_transform(df['State'])

# Split data into features (X) and target (y)
X = df.drop('Profit', axis=1)
y = df['Profit']

# Perform one-hot encoding on 'State' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate R-squared value
r2 = r2_score(y_test, y_pred)
print(f'R-squared value: {r2:.4f}')









################
##  2)  decision tree model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('50_Startups.csv')

label_encoder = LabelEncoder()  # Apply LabelEncoder to the 'State' column
df['State'] = label_encoder.fit_transform(df['State'])

results = pd.DataFrame(columns=['Model', 'R-squared']) # Create an empty DataFrame to store R-squared values for different models

X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']] # Define the features (X) and target variable (y)
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # Split the data into training and testing sets

model = DecisionTreeRegressor(random_state=0)  # Build a Decision Tree Regressor model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # Predict on the test set

r2 = r2_score(y_test, y_pred) # Calculate R-squared value for the Decision Tree Regressor model


results = results.append({'Model': 'Decision Tree Regression', 'R-squared': r2}, ignore_index=True) # Append the results to the DataFrame

r2 = r2_score(y_test, y_pred)  # Calculate R-squared value
print(f'R-squared value: {r2:.4f}')

# Print the table of results
print("\nTable of R-squared values:")
print(results)



####
## 3) Support Vector Regression (SVR) model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score


df = pd.read_csv('50_Startups.csv')

label_encoder = LabelEncoder()         # Apply LabelEncoder to the 'State' column
df['State'] = label_encoder.fit_transform(df['State'])
results = pd.DataFrame(columns=['Model', 'R-squared'])  # Create an empty DataFrame to store R-squared values for different models

X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]  # Define the features (X) and target variable (y)
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # Split the data into training and testing sets
scaler = StandardScaler()  # Standardize the features (important for SVR)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_model = SVR(kernel='linear')  # Build a Support Vector Regression (SVR) model
svr_model.fit(X_train_scaled, y_train)

y_pred = svr_model.predict(X_test_scaled) # Predict on the test set

r2 = r2_score(y_test, y_pred) # Calculate R-squared value for the SVR model

results = results.append({'Model': 'Support Vector Regression', 'R-squared': r2}, ignore_index=True)  # Append the results to the DataFrame

r2 = r2_score(y_test, y_pred)  # Calculate R-squared value
print(f'R-squared value: {r2:.4f}')

# Print the table of results
print("\nTable of R-squared values:")
print(results)



#################
### data visualization





import matplotlib.pyplot as plt

# Model names and corresponding R-squared values
models = ['Linear Regression', 'Decision Tree', 'Support Vector Regression']
r2_values = [0.9347, 0.9610, -0.1571]

# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(models, r2_values, color=['skyblue', 'lightgreen', 'coral'])

# Annotate the bars with R-squared values
for i, v in enumerate(r2_values):
    plt.text(v, i, f'{v:.4f}', color='black', va='center', fontsize=12)

# Set labels and title
plt.xlabel('R-squared Value')
plt.title('Comparison of R-squared Values for Regression Models')
plt.xlim(-1, 1)  # Set the x-axis limits for better visualization
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)  # Add a vertical line at x=0
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
















