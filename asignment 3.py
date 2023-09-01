


###  1) 
 
import numpy as np
from scipy import stats


# Data for Unit A and Unit B

import pandas as pd
df=pd.read_csv("Cutlets.csv")

df.shape
# Step 2: Assumption Testing

# Shapiro-Wilk test for normality
_, p_value_A = stats.shapiro(data_A)
_, p_value_B = stats.shapiro(data_B)

print("Shapiro-Wilk p-value for Unit A:", p_value_A)
print("Shapiro-Wilk p-value for Unit B:", p_value_B)

 
# Step 3: Decision and Conclusion

alpha = 0.05

if p_value_t < alpha:
    print("Reject the null hypothesis. There is a significant difference in cutlet diameter between Unit A and Unit B.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in cutlet diameter between Unit A and Unit B.")



# Step 4: Hypothesis Testing

# Perform two-sample t-test assuming equal variances
t_statistic, p_value_t = stats.ttest_ind(data_A, data_B, equal_var=True)

print("T-statistic:", t_statistic)
print("P-value:", p_value_t)

# Step 5: Decision and Conclusion

alpha = 0.05

if p_value_t < alpha:
    print("Reject the null hypothesis. There is a significant difference in cutlet diameter between Unit A and Unit B.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in cutlet diameter between Unit A and Unit B.")




##################


import scipy.stats as stats

# Data for the four laboratories


import pandas as pd
df=pd.read_csv("labTAT.csv")

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(lab1, lab2, lab3, lab4)

# Set the significance level
alpha = 0.05

# Print the results
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# Make a decision based on the p-value
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in TAT among the laboratories.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in TAT among the laboratories.")



################################

import scipy.stats as stats

# Observed values
observed = [[50, 142, 131, 70],
            [435, 1523, 1356, 750]]

# Perform the chi-squared test for independence
chi2, p, _, _ = stats.chi2_contingency(observed)

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print("Chi-squared statistic:", chi2)
print("P-value:", p)

# Check the p-value against alpha
if p < alpha:
    print("Reject the null hypothesis. There is a significant difference in male-female buyer ratios across regions.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in male-female buyer ratios across regions.")


####################################



import scipy.stats as stats

# Observed values
import pandas as pd
observed = pd.read_csv("BuyerRatio.csv")


# Perform the chi-squared test for independence
chi2, p, _, _ = stats.chi2_contingency(observed)

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print("Chi-squared statistic:", chi2)
print("P-value:", p)

# Check the p-value against alpha
if p < alpha:
    print("Reject the null hypothesis. There is a significant difference in male-female buyer ratios across regions.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in male-female buyer ratios across regions.")





#############


import scipy.stats as stats


# Observed values
observed = [[50, 142, 131, 70],
            [435, 1523, 1356, 750]]

# Perform the chi-squared test for independence
chi2, p, _, _ = stats.chi2_contingency(observed)

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print("Chi-squared statistic:", chi2)
print("P-value:", p)

# Check the p-value against alpha
if p < alpha:
    print("Reject the null hypothesis. There is a significant difference in male-female buyer ratios across regions.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in male-female buyer ratios across regions.")






####################



import numpy as np
from scipy import stats

# Create a contingency table (observed values)
data = [
    ["Error Free", "Error Free", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Defective"],
    ["Error Free", "Defective", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Defective", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Defective"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Error Free", "Defective"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Defective"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Defective", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Defective"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Error Free", "Defective", "Error Free", "Error Free"],
    ["Error Free", "Error Free", "Error Free", "Error Free"],
    ["Defective", "Error Free", "Error Free", "Error Free"]
]

# Create a contingency table (observed values)
observed = np.array([[row.count("Error Free") for row in data],
                     [row.count("Defective") for row in data]])

# Perform chi-squared test
chi2, p, _, _ = stats.chi2_contingency(observed)

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print("Chi-squared statistic:", chi2)
print("P-value:", p)



######################

import numpy as np
from scipy import stats

# Create a contingency table (observed values)
import pandas as pd
observed=pd.read_csv("Costomer+OrderForm.csv")
# Create a contingency table (observed values)
observed = np.array([[row.count("Error Free") for row in data],
                     [row.count("Defective") for row in data]])

# Perform chi-squared test
chi2, p, _, _ = stats.chi2_contingency(observed)

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print("Chi-squared statistic:", chi2)
print("P-value:", p)






#############################

import pandas as pd
observed = pd.read_csv("Costomer+OrderForm.csv")

# Perform chi-squared test
chi2, p, _, _ = stats.chi2_contingency(observed)



#######################

import pandas as pd
from scipy.stats import chi2_contingency

# Create a dictionary to represent the data
data = {
    'Phillippines': ['Error Free'] * 50 + ['Defective'] * 10,
    'Indonesia': ['Error Free'] * 50 + ['Defective'] * 10,
    'Malta': ['Error Free'] * 50 + ['Defective'] * 10,
    'India': ['Error Free'] * 50 + ['Defective'] * 10
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Create a contingency table
contingency_table = pd.crosstab(df['Phillippines'], [df['Indonesia'], df['Malta'], df['India']])

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print("Chi-squared statistic:", chi2)
print("P-value:", p)
p.round(2)

# Check the p-value against alpha
if p < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis ")




#############################

                Philippines   Indonesia   Malta   India
Error Free       173           175       128    169
Defective         10            10        32     13


import scipy.stats as stats

# Create the contingency table
observed = [[173, 175, 128, 169],
            [10, 10, 32, 13]]

# Perform the chi-squared test
chi2, p, _, _ = stats.chi2_contingency(observed)

# Print the results
print("Chi-squared statistic:", chi2)
print("P-value:", p)

































































































































