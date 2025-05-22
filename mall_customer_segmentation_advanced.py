import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the Dataset
df = pd.read_csv("Mall_Customers.csv")


data = df.drop(['CustomerID', 'Gender'], axis=1)

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


silhouette_scores = []
K_range = range(2, 11)  # silhouette needs k>=2

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8,4))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters according to Silhouette Score: {optimal_k}")


kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(scaled_data)
df['Cluster'] = labels

colors = plt.cm.get_cmap('tab10', optimal_k)

# a) Plot clusters on Annual Income vs Spending Score with centroids
plt.figure(figsize=(8,6))
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                color=colors(cluster), label=f'Cluster {cluster}', s=60)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:,1], centroids[:,2], s=300, c='black', marker='X', label='Centroids')
plt.title('Customer Clusters with Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

features = data.columns.tolist()
num_features = len(features)
plt.figure(figsize=(15, 15))
for i in range(num_features):
    for j in range(num_features):
        plt.subplot(num_features, num_features, i*num_features + j + 1)
        if i == j:
            plt.text(0.5, 0.5, features[i], fontsize=9, ha='center', va='center')
            plt.xticks([])
            plt.yticks([])
        else:
            for cluster in range(optimal_k):
                cluster_data = df[df['Cluster'] == cluster]
                plt.scatter(cluster_data[features[j]], cluster_data[features[i]],
                            color=colors(cluster), s=10, alpha=0.5)
            if i == num_features-1:
                plt.xlabel(features[j], fontsize=7)
            else:
                plt.xticks([])
            if j == 0:
                plt.ylabel(features[i], fontsize=7)
            else:
                plt.yticks([])
plt.suptitle('Pairwise Feature Scatter Plots by Cluster', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


print("\n--- Business Insights by Cluster ---")
cluster_summary = df.groupby('Cluster')[features].mean().round(2)
cluster_sizes = df['Cluster'].value_counts().sort_index()
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} (Size: {cluster_sizes[cluster]} customers):")
    print(cluster_summary.loc[cluster])
    # Example insight
    if cluster_summary.loc[cluster]['Spending Score (1-100)'] > 60:
        print("-> This cluster consists of high spenders.")
    elif cluster_summary.loc[cluster]['Annual Income (k$)'] > cluster_summary['Annual Income (k$)'].mean():
        print("-> This cluster has higher income customers but average spending.")
    else:
        print("-> This cluster has lower income and spending.")


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))
for cluster in range(optimal_k):
    plt.scatter(pca_result[df['Cluster'] == cluster, 0], pca_result[df['Cluster'] == cluster, 1],
                color=colors(cluster), label=f'Cluster {cluster}', s=70)
plt.title('Clusters Visualized Using PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
