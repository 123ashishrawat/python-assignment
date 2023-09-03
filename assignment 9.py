# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:05:27 2023

@author: India
"""

pip install pandas mlxtend
import pandas as pd

data = pd.read_csv('book.csv')


from mlxtend.frequent_patterns import apriori, association_rules

# Define different support and confidence values to test
support_values = [0.1, 0.2, 0.3]
confidence_values = [0.5, 0.6, 0.7]

# Iterate through different support and confidence values
for support in support_values:
    for confidence in confidence_values:
        # Generate frequent itemsets
        frequent_itemsets = apriori(data, min_support=support, use_colnames=True)

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)

        # Print support, confidence, and the number of rules
        print(f"Support: {support}, Confidence: {confidence}")
        print(f"Number of Rules: {len(rules)}")
        print(rules)
        
        
   ####### data visualisation

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame with the provided dataset
data = pd.read_csv('book.csv')

# Generate frequent itemsets
frequent_itemsets = apriori(data, min_support=0.2, use_colnames=True)

# Generate association rules with lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Visualization of rules

# 1. Scatter plot for support vs. confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(x="support", y="confidence", data=rules)
plt.title("Support vs. Confidence")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()

# 2. Scatter plot for support vs. lift
plt.figure(figsize=(8, 6))
sns.scatterplot(x="support", y="lift", data=rules)
plt.title("Support vs. Lift")
plt.xlabel("Support")
plt.ylabel("Lift")
plt.show()

# 3. Scatter plot for confidence vs. lift
plt.figure(figsize=(8, 6))
sns.scatterplot(x="confidence", y="lift", data=rules)
plt.title("Confidence vs. Lift")
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.show()
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
