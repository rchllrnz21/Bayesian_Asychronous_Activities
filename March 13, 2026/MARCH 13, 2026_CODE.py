# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:46:05 2026

@author: rochelle
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv(r"C:/Users/rochelle/Downloads/LORENZO_SAMPLE_UNSUPERVISED_Movies_Dataset.csv")

# Drop rows with missing values (your CSV has many empty trailing rows)
df = df.dropna()

# Select features for clustering
features = ["Year", "Duration_Minutes", "Genre", "Director", "Language", 
            "Country", "IMDb_Rating", "Votes_Thousands", 
            "Budget_Million_USD", "BoxOffice_Million_USD", "Age_Rating"]

X = df[features].copy()

# Encode categorical features
categorical_cols = ["Genre", "Director", "Language", "Country", "Age_Rating"]
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display the full table with clusters
print("\n=== Full Movies Table with Clusters ===")
print(df[["Movie_ID", "Title", "Year", "Genre", "Director", "IMDb_Rating", "Cluster"]].to_string(index=False))

# Display cluster counts summary
print("\n=== Cluster Summary ===")
print(df["Cluster"].value_counts().sort_index())

# Optional: Show cluster centers (scaled space)
print("\n=== Cluster Centers (scaled) ===")
print(kmeans.cluster_centers_)
