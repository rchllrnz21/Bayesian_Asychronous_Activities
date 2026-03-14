# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:51:01 2026

@author: rochelle
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:/Users/rochelle/Downloads/LORENZO_SAMPLE_UNSUPERVISED_Movies_Dataset.csv")

# Drop rows with missing values
df = df.dropna()

# Select numeric features for clustering (exclude non-numeric columns)
X = df[['Year',
        'Duration_Minutes',
        'IMDb_Rating',
        'Votes_Thousands',
        'Budget_Million_USD',
        'BoxOffice_Million_USD']]

# Scale features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Show results with movie titles
results = df[["Title", "Director", "Cluster", "IMDb_Rating", "BoxOffice_Million_USD"]]

# Print the full table
print("\nClustering Results (all rows):")
print(results.to_string(index=False))

# Save to CSV
results.to_csv("clustering_output.csv", index=False)
print("\nTable saved as 'clustering_output.csv'")

# -----------------------------
# Plot the clusters using PCA
# -----------------------------

# Reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot of clusters
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["Cluster"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Movie Clusters Visualization")
plt.show()