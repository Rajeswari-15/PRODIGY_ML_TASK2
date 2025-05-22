import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

  
df = pd.read_csv("Mall_Customers.csv")


data = df.drop(['CustomerID', 'Gender'], axis=1)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)
df['Cluster'] = kmeans.labels_


plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                color=colors[cluster], label=f'Cluster {cluster}', s=70)

plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
for cluster in range(5):
    plt.scatter(pca_result[df['Cluster'] == cluster, 0], 
                pca_result[df['Cluster'] == cluster, 1], 
                color=colors[cluster], label=f'Cluster {cluster}', s=70)

plt.title('Clusters visualized using PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()


print("\nCluster Summary (Average values):")
print(df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())

df.to_csv("Mall_Customers_with_Clusters.csv", index=False)
print("Clustered data saved to 'Mall_Customers_with_Clusters.csv'")