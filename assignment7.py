# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 07:35:54 2023

@author: India
"""
### problem statement 1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

data=pd.read_csv("EastWestAirlines.csv") 


# Feature selection (excluding 'ID#' and 'Award?' columns)
X = data.iloc[:, 1:-1]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical Clustering
plt.figure(figsize=(12, 6))
dendrogram(linkage(X_scaled, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# K-means Clustering with Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for K-means')
plt.show()

# K-means Clustering with Optimal Number of Clusters
optimal_k = 3  # You can choose the number based on the Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)
 
 
print(data.columns)

# Choose two features (columns) for the scatter plot
x_feature = 'Balance' 
y_feature = 'Flight_miles_12mo'   

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data[x_feature], data[y_feature], c=kmeans_labels, cmap='viridis', s=50)  # Adjust 's' for marker size

# Add labels and title
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title('K-means Clustering')

# Show the colorbar to map cluster colors
cbar = plt.colorbar()
cbar.set_label('Cluster')

# Show the plot
plt.show()


# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate K-means Clustering with Silhouette Score
silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
print(f'Silhouette Score for K-means: {silhouette_avg}')

# Evaluate DBSCAN Clustering
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f'Number of clusters (DBSCAN): {n_clusters_dbscan}')









### problem statement 2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the crime data
data=pd.read_csv("crime_data.csv")

# Extract the state names (you can use it later for labeling)
states = data['States']

# Encode the state names into numeric labels
label_encoder = LabelEncoder()
data['State_Label'] = label_encoder.fit_transform(states)

# Drop the non-numeric columns (States and State_Label) before standardization
data = data.drop(columns=['States', 'State_Label'])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Hierarchical Clustering
plt.figure(figsize=(12, 6))
dendrogram(linkage(data_scaled, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# K-means Clustering with Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for K-means')
plt.show()

# K-means Clustering with Optimal Number of Clusters
optimal_k = 5  # You can choose the number based on the Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans_labels = kmeans.fit_predict(data_scaled)



# Create a scatter plot
plt.figure(figsize=(8, 6))

# we can choose different features for the x and y axes
x_feature = 'Murder'
y_feature = 'Assault'

# Use the cluster labels for color coding
plt.scatter(data[x_feature], data[y_feature], c=kmeans_labels, cmap='viridis', s=50)  # Adjust 's' for marker size

# Add labels and title
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title('K-means Clustering')

# Show the colorbar to map cluster colors
cbar = plt.colorbar()
cbar.set_label('Cluster')

# Show the plot
plt.show()







# DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=3).fit(data_scaled)

# Add cluster labels back to the original data
data['Hierarchical_Cluster'] = AgglomerativeClustering(n_clusters=optimal_k).fit_predict(data_scaled)
data['KMeans_Cluster'] = kmeans_labels
data['DBSCAN_Cluster'] = dbscan.labels_

# Analyze the clusters
cluster_means = data.groupby('KMeans_Cluster').mean()
cluster_counts = data['KMeans_Cluster'].value_counts()

# Print cluster means and sizes for K-means clustering
print("Cluster Means (K-means):")
print(cluster_means)

print("\nCluster Sizes (K-means):")
print(cluster_counts)

# Visualize the clusters if desired (e.g., Murder vs. Assault)
plt.scatter(data['Murder'], data['Assault'], c=kmeans_labels, cmap='viridis')
plt.xlabel('Murder Rate')
plt.ylabel('Assault Rate')
plt.title('K-means Clustering')
plt.show()
















