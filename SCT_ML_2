#AUTHOR : NATHISHWAR
#DATASET : DOWNLOAD FROM KAGGLE (MALL_CUSTOMERS.CSV)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = '/mnt/data/Mall_Customers.csv'
df = pd.read_csv(data_path)

# Display basic info
display(df.head())
display(df.info())

# Select relevant features for clustering (e.g., Annual Income and Spending Score)
X = df.iloc[:, [3, 4]].values  # Adjust column indices if needed

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Train K-Means with optimal clusters (let's assume 5 from the elbow plot)
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters  # Assign clusters to original data

# Visualizing clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
            kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
            s=300, c='red', label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Save clustered data
df.to_csv('/mnt/data/Mall_Customers_Clustered.csv', index=False)
