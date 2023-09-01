# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:56:56 2023

@author: India
"""

import numpy as np
import matplotlib.pyplot as plt

# Data (in percentage format)
data = [24.23, 25.53, 25.41, 24.14, 29.62, 28.25, 25.81, 24.39, 40.26, 32.95, 91.36, 25.99, 39.42, 26.71, 35.00]

# Convert percentages to decimal values
data = [float(x.strip('%')) / 100 for x in data]

# Plot the data
plt.figure(figsize=(10, 5))
plt.boxplot(data, vert=False)
plt.title("Box Plot of Data")
plt.xlabel("Measure X")
plt.show()

# Calculate mean, standard deviation, and variance
mean = np.mean(data)
std_dev = np.std(data)
variance = np.var(data)

print(f"Mean (μ): {mean:.4f}")
print(f"Standard Deviation (σ): {std_dev:.4f}")
print(f"Variance (σ^2): {variance:.4f}")

# Identify potential outliers using the IQR method
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = [x for x in data if x < lower_bound or x > upper_bound]

print("Potential Outliers:")
for outlier in outliers:
    print(outlier)
    
    
    2)
    
    import matplotlib.pyplot as plt

# Data
data = [0, 5, 7.3, 12.6, 19]

# Create a box plot
plt.boxplot(data, vert=False, whis=[0, 100], sym='ro', widths=0.7)

# Add labels and title
plt.xlabel("Data")
plt.title("Box Plot")

# Show the plot
plt.show()



###############


import scipy.stats as stats

# Parameters
mean = 45  # Mean time in minutes
std_dev = 8  # Standard deviation in minutes
time_allowed = 60  # Time allowed in minutes
head_start = 10  # Time before servicing begins in minutes

# Calculate the z-score for the adjusted time
z_score = (time_allowed - mean - head_start) / std_dev

# Calculate the probability using the CDF of the standard normal distribution
probability = 1 - stats.norm.cdf(z_score)

print(f"The probability that the service manager cannot meet his commitment is approximately: {probability:.4f}")


##########################

import scipy.stats as stats

# Parameters
mean = 38
std_dev = 6
total_employees = 400

# A. Probability of employees older than 44
prob_older_than_44 = 1 - stats.norm.cdf(44, loc=mean, scale=std_dev)

# B. Probability of employees under the age of 30
prob_under_30 = stats.norm.cdf(30, loc=mean, scale=std_dev)

# Calculate the expected number of employees under 30
expected_employees_under_30 = prob_under_30 * total_employees

# Evaluate the statements
statement_A_true = prob_older_than_44 > stats.norm.cdf(38, loc=mean, scale=std_dev)
statement_B_true = round(expected_employees_under_30) == 36

print("Statement A (More employees older than 44 than between 38 and 44):", statement_A_true)
print("Statement B (Training program for employees under 30 attracts about 36 employees):", statement_B_true)


#################

import scipy.stats as stats

# Parameters of the normal distribution
mean = 100
std_deviation = 20

# Percentiles for the desired probability range
percentile_low = 0.005
percentile_high = 0.995

# Calculate the z-scores for the percentiles
z_low = stats.norm.ppf(percentile_low)
z_high = stats.norm.ppf(percentile_high)

# Calculate the values of a and b
a = mean + z_low * std_deviation
b = mean + z_high * std_deviation

print("a =", a)
print("b =", b)



###########################

import scipy.stats as stats

# Given values
mean_profit1 = 5  # Mean profit for division 1 (in million dollars)
std_deviation_profit1 = 3  # Standard deviation for division 1 (in million dollars)

mean_profit2 = 7  # Mean profit for division 2 (in million dollars)
std_deviation_profit2 = 4  # Standard deviation for division 2 (in million dollars)

# Conversion rate from dollars to Rupees
conversion_rate = 45

# A. Rupee range for 95% probability (centered on the mean)
total_mean = mean_profit1 + mean_profit2
total_std_deviation = (std_deviation_profit1**2 + std_deviation_profit2**2)**0.5

z_2_5 = stats.norm.ppf(0.025)
z_97_5 = stats.norm.ppf(0.975)

lower_bound_dollars = total_mean - z_97_5 * total_std_deviation
upper_bound_dollars = total_mean + z_97_5 * total_std_deviation

lower_bound_rupees = lower_bound_dollars * conversion_rate
upper_bound_rupees = upper_bound_dollars * conversion_rate

print(f"A. Rupee range for 95% probability (centered on the mean): {lower_bound_rupees:.2f} Rupees to {upper_bound_rupees:.2f} Rupees")

# B. 5th percentile of profit in Rupees
z_5 = stats.norm.ppf(0.05)

percentile_5_dollars = total_mean + z_5 * total_std_deviation
percentile_5_rupees = percentile_5_dollars * conversion_rate

print(f"B. 5th percentile of profit in Rupees: {percentile_5_rupees:.2f} Rupees")

# C. Probability of making a loss for each division
z_zero_profit1 = (0 - mean_profit1) / std_deviation_profit1
probability_loss1 = stats.norm.cdf(z_zero_profit1)

z_zero_profit2 = (0 - mean_profit2) / std_deviation_profit2
probability_loss2 = stats.norm.cdf(z_zero_profit2)

if probability_loss1 > probability_loss2:
    print("C. Division 1 is more likely to make a loss in a given year.")
elif probability_loss2 > probability_loss1:
    print("C. Division 2 is more likely to make a loss in a given year.")
else:
    print("C. Both divisions have the same probability of making a loss in a given year.")


##################

import math

# Desired margin of error (as a decimal)
margin_of_error = 0.04

# Confidence level (as a decimal)
confidence_level = 0.95

# Z-score for the given confidence level (95% confidence corresponds to 1.96)
z_score = 1.96

# Estimated proportion (use 0.5 for maximum variability if no estimate is available)
p_hat = 0.5

# Calculate the minimum sample size
minimum_sample_size = math.ceil((z_score**2 * p_hat * (1 - p_hat)) / margin_of_error**2)

print(f"Minimum sample size required: {minimum_sample_size}")



import math

# Desired margin of error (as a decimal)
margin_of_error = 0.04

# Confidence level (as a decimal)
confidence_level = 0.98

# Z-score for the given confidence level (98% confidence corresponds to 2.33)
z_score = 2.33

# Estimated proportion (use 0.5 for maximum variability if no estimate is available)
p_hat = 0.5

# Calculate the minimum sample size
minimum_sample_size = math.ceil((z_score**2 * p_hat * (1 - p_hat)) / margin_of_error**2)

print(f"Minimum sample size required: {minimum_sample_size}")


# Define the candies and their respective probabilities
candies = [1, 4, 3, 5, 6, 2]
probabilities = [0.015, 0.20, 0.65, 0.005, 0.01, 0.120]

# Calculate the expected number of candies
expected_candies = sum([x * p for x, p in zip(candies, probabilities)])

# Print the result
print("Expected number of candies:", expected_candies)



#################

# List of weights (in pounds) of patients
weights = [108, 110, 123, 134, 135, 145, 167, 187, 199]

# Calculate the expected value (mean)
expected_value = sum(weights) / len(weights)

# Print the result
print("Expected Value of Weight:", expected_value)



######################


import scipy.stats as stats

# Given data
mu = 50  # Population mean
sigma = 40  # Population standard deviation
n = 100  # Sample size
lower_limit = 45
upper_limit = 55

# Calculate the standard error
se = sigma / (n**0.5)

# Calculate the z-scores for the lower and upper limits
z_lower = (lower_limit - mu) / se
z_upper = (upper_limit - mu) / se

# Calculate the cumulative probabilities
prob_lower = stats.norm.cdf(z_lower)
prob_upper = stats.norm.cdf(z_upper)

# Calculate the probability of an investigation
probability_investigation = 1 - (prob_upper - prob_lower)

# Convert probability to percentage
percentage_investigation = probability_investigation * 100

print(f"Probability of an investigation: {percentage_investigation:.2f}%")


###############################################

import scipy.stats as stats

mean = 50
std_dev = 4

lower_threshold = (45 - mean) / std_dev
upper_threshold = (55 - mean) / std_dev

probability = stats.norm.cdf(upper_threshold) - stats.norm.cdf(lower_threshold)

print("Probability of investigation:", probability)



target_probability = 0.05

z_score = stats.norm.ppf(1 - target_probability/2)  # Two-tailed distribution

required_sample_size = ((z_score * std_dev) / (5)) ** 2

print("Minimum number of transactions to sample:", required_sample_size)





























