# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 08:23:37 2023

@author: India
"""
import numpy as np   # Import necessary libraries
import pandas as pd   
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd                         # Create a DataFrame with the given data
data=pd.read_csv("Salary_Data.csv")
df = pd.DataFrame(data)
plt.scatter(df['YearsExperience'], df['Salary'])  # EDA - Explore the data
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs. Salary')
plt.show()
# Create the feature matrix (X) and target variable (y) # Split the data into training and testing sets (optional)
X = df[['YearsExperience']]  # from sklearn.model_selection import train_test_split
y = df['Salary']                    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()            # Build the linear regression model
model.fit(X, y) 
predictions = model.predict(X)  # Make predictions
mse = mean_squared_error(y, predictions)   # Calculate the Mean Squared Error (MSE)
print(f"Mean Squared Error: {mse}")

import numpy as np
rmse = np.sqrt(mse)
print("Root Mean squarred error: ",rmse.round(2))


plt.scatter(df['YearsExperience'], df['Salary'])
plt.plot(df['YearsExperience'], predictions, color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs. Salary (Regression Line)')  # Visualize the regression line
plt.show()

# Predict Salary for a specific Years of Experience
years_of_experience = 13
predicted_salary = model.predict([[years_of_experience]])
print(f"Predicted Salary for {years_of_experience} years of experience: ${predicted_salary[0]:.2f}")




#########################



import numpy as np  # Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=pd.read_csv("delivery_time.csv") # Create a DataFrame with the given data
df = pd.DataFrame(data)
plt.scatter(df['SortingTime'], df['DeliveryTime'])  # EDA - Explore the data
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Sorting Time vs. Delivery Time')
plt.show()

X = df[['SortingTime']]  # Create the feature matrix (X) and target variable (y)
y = df['DeliveryTime']
model = LinearRegression()  # Build the linear regression model
model.fit(X, y)

predictions = model.predict(X)   # Make predictions

mse = mean_squared_error(y, predictions)  # Calculate the Mean Squared Error (MSE)
print(f"Mean Squared Error: {mse}")

plt.scatter(df['SortingTime'], df['DeliveryTime'])   # Visualize the regression line
plt.plot(df['SortingTime'], predictions, color='red')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Sorting Time vs. Delivery Time (Regression Line)')
plt.show()

sorting_time = 12     # Predict Delivery Time for a specific Sorting Time
predicted_delivery_time = model.predict([[sorting_time]])
print(f"Predicted Delivery Time for Sorting Time {sorting_time}: {predicted_delivery_time[0]:.2f} hours")



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Create a DataFrame with the provided data

data=pd.read_csv("delivery_time.csv") # Create a DataFrame with the given data
df = pd.DataFrame(data)

# Define a function to build and evaluate regression models with different transformations
def build_and_evaluate_model(data, x_col, y_col, transformation_name):
    # Apply the specified transformation to the x variable
    if transformation_name == 'log':
        data[x_col] = np.log(data[x_col])
    elif transformation_name == 'square':
        data[x_col] = np.square(data[x_col])
    elif transformation_name == 'sqrt':
        data[x_col] = np.sqrt(data[x_col])

    # Split the data into features (X) and target (y)
    X = data[[x_col]]
    y = data[y_col]

    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the delivery time
    y_pred = model.predict(X)

    # Calculate the RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return model, rmse

# List of transformation names
transformations = ['none', 'log', 'square', 'sqrt']

# Create a DataFrame to store RMSE values for each transformation
rmse_results = pd.DataFrame(columns=['Transformation', 'RMSE'])

# Build models with different transformations and calculate RMSE values
for transformation in transformations:
    if transformation == 'none':
        x_col_name = 'Sorting Time (Original)'
    else:
        x_col_name = f'Sorting Time ({transformation.capitalize()})'
    model, rmse = build_and_evaluate_model(data.copy(), 'Sorting Time', 'Delivery Time', transformation)
    rmse_results = rmse_results.append({'Transformation': x_col_name, 'RMSE': rmse}, ignore_index=True)

# Display RMSE results
print(rmse_results)

# Plot RMSE values for different transformations
plt.figure(figsize=(8, 6))
plt.bar(rmse_results['Transformation'], rmse_results['RMSE'])
plt.xlabel('Transformation')
plt.ylabel('RMSE')
plt.title('RMSE for Different Transformations')
plt.show()













