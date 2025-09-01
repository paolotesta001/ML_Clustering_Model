"""
Created on Tue may 10 19:39:11 2025

@author: Utente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist
import matplotlib.cm as cm

# Load the dataset
dataset = pd.read_csv('penguins_lter.csv')

# Drop unnecessary columns
drop_cols = ['studyName', 'Sample Number', 'Individual ID', 'Date Egg', 'Comments']
dataset = dataset.drop(columns=[col for col in drop_cols if col in dataset.columns])

# Choose categorical columns to encode
categorical_cols = [col for col in ['Region', 'Island', 'Stage'] if col in dataset.columns]

# One-hot encode categorical columns
data_enc = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True, dummy_na=True)

# Update feature columns list
exclude_cols = ['Species', 'Clutch Completion', 'Sex']
all_feature_cols = [col for col in data_enc.columns if col not in exclude_cols]

# Handle missing values (drop rows with missing data)
data_enc = data_enc.dropna(subset=all_feature_cols + ['Species'])

# Prepare feature matrix and target vector
features = data_enc[all_feature_cols].astype(float)
species = data_enc['Species'].values

# Update list of numeric columns after encoding
numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# KMeans: Elbow method to find optimal number of clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow method to choose k')
plt.show()

# Choose number of clusters
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
centroids = kmeans.cluster_centers_

# Assign clusters and true species
df_results = pd.DataFrame(features, columns=all_feature_cols)
df_results['cluster'] = clusters
df_results['Species'] = species

print("\nCluster distribution:")
print(df_results['cluster'].value_counts())

# Calculate purity
contingency = pd.crosstab(df_results['cluster'], df_results['Species'])
purity = contingency.max(axis=1).sum() / contingency.sum().sum()
print(f"\nPurity: {purity:.4f}")

print("\nSpecies distribution within clusters:")
print(df_results.groupby('cluster')['Species'].value_counts(normalize=True))

# Distance between centroids
dist_matrix = cdist(centroids, centroids, metric='euclidean')
print("\nDistance matrix between cluster centroids:")
print(dist_matrix)

# Choose two numeric features to visualize
numeric_encoded_cols = [col for col in features.columns if col in numeric_cols]
if len(numeric_encoded_cols) >= 2:
    x_var = numeric_encoded_cols[0]
    y_var = numeric_encoded_cols[1]
else:
    x_var, y_var = features.columns[:2]

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=df_results[x_var],
    y=df_results[y_var],
    hue=df_results['cluster'],
    palette="tab10",
    alpha=0.7
)
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.title(f'Clusters visualized on {x_var} vs {y_var}')
plt.legend(title='Cluster')
plt.show()

# Visualize centroids on the same plot
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=df_results[x_var],
    y=df_results[y_var],
    hue=df_results['cluster'],
    palette="tab10",
    alpha=0.7
)
# Centroids are inverse-transformed to original feature space for plotting
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features.columns)
plt.scatter(
    centroids_df[x_var], 
    centroids_df[y_var], 
    c='black', s=200, marker='X', label='Centroids'
)
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.title(f'Clusters visualized on {x_var} vs {y_var} with centroids')
plt.legend(title='Cluster')
plt.show()

# Silhouette score
silhouette_avg = silhouette_score(features_scaled, clusters)
silhouette_vals = silhouette_samples(features_scaled, clusters)
print(f"\nAverage silhouette score: {silhouette_avg:.4f}")

y_lower = 10
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    ith_cluster_silhouette_values = silhouette_vals[clusters == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Average silhouette = {silhouette_avg:.2f}")
plt.xlabel("Silhouette value")
plt.ylabel("Sample index")
plt.title(f"Silhouette plot for {n_clusters} clusters")
plt.legend()
plt.show()

# Calculate purity
contingency = pd.crosstab(df_results['cluster'], df_results['Species'])
purity = contingency.max(axis=1).sum() / contingency.sum().sum()
print(f"\nPurity: {purity:.4f}")
