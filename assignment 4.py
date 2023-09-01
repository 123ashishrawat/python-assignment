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


