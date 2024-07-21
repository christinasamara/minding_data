import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def read_dataset():
    li = []
    for filename in glob.glob("harth\*"):
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.drop(['index', 'Unnamed: 0'], axis=1)

    frame['timestamp'] = pd.to_datetime(df['timestamp'])
    frame['time_diff'] = frame['timestamp'].diff().dt.total_seconds()
    frame = frame.drop(['timestamp'], axis=1)
    frame['time_diff'] = frame['time_diff'].fillna(0)

    print(frame.groupby('label').describe())
    frame.info()
    
    #mean_df = frame.groupby('label').mean(numeric_only=True)
    #stdev_df = frame.groupby('label').std()
    #print(mean_df)
    #print(stdev_df)
    return frame

frame = read_dataset()
frame = frame.drop(['label'], axis=1)

num_bins = 10
for column in frame.select_dtypes(include=['float64']):
    frame[column] = pd.qcut(frame[column], q=num_bins, labels=False, duplicates='drop')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(frame)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
print("PCA done")

n_clusters = 8

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
print("KMeans started")
clusters_kmeans = kmeans.fit_predict(X_pca)
print("KMeans fit_predict done")

minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
print("MiniBatchKMeans started")
clusters_minibatch = minibatch_kmeans.fit_predict(X_pca)
print("MiniBatchKMeans fit_predict done")


np.savetxt('cluster_labels_kmeans.csv', clusters_kmeans, delimiter=',', fmt='%d')
np.savetxt('cluster_labels_minibatch.csv', clusters_minibatch, delimiter=',', fmt='%d')
np.savetxt('cluster_centers_kmeans.csv', pca.inverse_transform(kmeans.cluster_centers_), delimiter=',')
np.savetxt('cluster_centers_minibatch.csv', pca.inverse_transform(minibatch_kmeans.cluster_centers_), delimiter=',')


# 3_clustering_comparison.py 
centers_kmeans = np.loadtxt('cluster_centers_kmeans.csv', delimiter=',')
centers_minibatch = np.loadtxt('cluster_centers_minibatch.csv', delimiter=',')

pca = PCA(n_components=3)
centers_kmeans_3d = pca.fit_transform(centers_kmeans)
centers_minibatch_3d = pca.transform(centers_minibatch)

# COMPARISON
sum_distances_kmeans = 0
for i, center in enumerate(centers_kmeans_3d):
    cluster_points = X_pca[clusters_kmeans == i]
    distances = euclidean_distances(cluster_points, [center])
    sum_distances_kmeans += np.sum(distances)

sum_distances_minibatch = 0
for i, center in enumerate(centers_minibatch_3d):
    cluster_points = X_pca[clusters_minibatch == i]
    distances = euclidean_distances(cluster_points, [center])
    sum_distances_minibatch += np.sum(distances)

print(f'Sum of Distances for KMeans: {sum_distances_kmeans}')
print(f'Sum of Distances for MiniBatchKMeans: {sum_distances_minibatch}')

avg_distance_kmeans = sum_distances_kmeans / len(X_pca)
avg_distance_minibatch = sum_distances_minibatch / len(X_pca)

print(f'Average Distance to Cluster Center for KMeans: {avg_distance_kmeans}')
print(f'Average Distance to Cluster Center for MiniBatchKMeans: {avg_distance_minibatch}')





# SCATTER PLOT
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters_kmeans, cmap='viridis', marker='o')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('KMeans Clusters')
fig.colorbar(scatter, ax=ax, pad=0.2, label="Cluster Labels")

ax = fig.add_subplot(122, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters_minibatch, cmap='viridis', marker='o')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('MiniBatchKMeans Clusters')
fig.colorbar(scatter, ax=ax, pad=0.2, label="Cluster Labels")
plt.tight_layout()
plt.savefig('clusters_comparison.png')
plt.close('all')


# HEATMAP
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
kmeans_cluster_centers = pca.inverse_transform(kmeans.cluster_centers_)
minibatch_cluster_centers = pca.inverse_transform(minibatch_kmeans.cluster_centers_)
sns.heatmap(kmeans_cluster_centers, ax=axes[0], cmap='viridis')
axes[0].set_title('KMeans Cluster Centers')
axes[0].set_xlabel('Feature Index')
axes[0].set_ylabel('Cluster Labels')
sns.heatmap(minibatch_cluster_centers, ax=axes[1], cmap='viridis')
axes[1].set_title('MiniBatchKMeans Cluster Centers')
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Cluster Labels')
plt.tight_layout()
plt.savefig('kmeans_vs_minibatch_heatmaps.png')