# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 10:45:41 2023

@author: India
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy

data = pd.read_csv('wine.csv')

# Extract features (exclude class column)
X = data.drop(columns=['Type'])

# Perform PCA to reduce dimensions to 3 principal components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X_pca)

# K-means clustering with scree plot for finding the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    sse.append(kmeans.inertia_)

# Plot the scree plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Scree Plot for K-means Clustering')
plt.show()

# Based on the scree plot, choose the optimal number of clusters (e.g., 3 clusters)
optimal_num_clusters = 3

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

# Compare cluster labels from hierarchical clustering and K-means with original class labels (Type)
original_labels = data['Type']

# Print cluster assignments and original class labels
print("Hierarchical Clustering Labels:")
print(agg_labels)
print("\nK-means Clustering Labels:")
print(kmeans_labels)
print("\nOriginal Class Labels:")
print(original_labels)









