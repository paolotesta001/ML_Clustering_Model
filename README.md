# 🐧 Penguin Species Clustering with KMeans

This project applies unsupervised machine learning techniques to cluster penguin species based on the Palmer Penguins dataset. Using KMeans clustering, feature engineering, and evaluation metrics, we analyze how effectively the algorithm can separate species without the need for labeled data.

# 📌 Project Overview
This analysis follows a structured four-step process:

Data Preprocessing: The dataset was prepared by dropping irrelevant columns, handling missing values, encoding categorical variables, and normalizing features to ensure consistency.

KMeans Clustering: The KMeans algorithm was applied. The optimal number of clusters (k) was determined using the Elbow Method to identify the point of diminishing returns in the sum of squared distances.

Evaluation: The performance of the clustering was evaluated using key metrics:

Purity Score: To measure how well our clusters align with the true penguin species labels.

Silhouette Score: To assess the cohesion within each cluster and the separation between different clusters.

Visualization: We visualized the results using scatterplots, centroid plots, and silhouette diagrams to provide a clear understanding of the cluster distributions and quality.

# ⚙️ Tech Stack
This project was built using the following Python libraries, a standard stack for data science and machine learning:

Pandas & NumPy: Essential for data preprocessing, manipulation, and numerical operations.

Matplotlib & Seaborn: Used for creating all the data visualizations.

Scikit-learn: The core library for implementing KMeans, data scaling, and calculating clustering metrics.

SciPy: Utilized for distance calculations, such as the cdist function.

# 📊 Key Results
The unsupervised KMeans model successfully identified distinct clusters that closely align with the known penguin species.

Optimal Number of Clusters: 3
The Elbow Method analysis clearly indicated that 3 clusters was the optimal number, which precisely matches the three species of penguins in the dataset (Adelie, Chinstrap, and Gentoo).

<img width="153" height="83" alt="image" src="https://github.com/user-attachments/assets/4e2aee3e-739b-4801-9079-3447acde7bfc" />
<img width="444" height="102" alt="image" src="https://github.com/user-attachments/assets/9f3f5d88-a21e-4be7-b8c8-7f5f321c9cb0" />


Purity Score: 0.92
A purity score of approximately 0.92 demonstrates a very strong alignment between the generated clusters and the true species, indicating the model's high effectiveness.

<img width="109" height="25" alt="image" src="https://github.com/user-attachments/assets/cb70104e-229e-4aea-9b6b-c2f1b1069bdd" />

Silhouette Score
The silhouette score further confirmed good cluster separation and compactness, proving the model was successful in its task.

<img width="228" height="21" alt="image" src="https://github.com/user-attachments/assets/e99cde90-1742-45a4-9a49-9601bdf64f8c" />


# 📈 Visualizations
1. Elbow Method
This plot shows the sum of squared distances for each number of clusters. The "elbow" at k=3 is a clear indicator of the optimal number of clusters.

<img width="800" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/d9595dce-b370-405a-ba88-23b3fb2bba9e" />



2. Cluster Plot with Centroids
This scatterplot visualizes the data points grouped into their respective clusters, with the cluster centroids highlighted as black 'X' markers.

<img width="422" height="272" alt="image" src="https://github.com/user-attachments/assets/44cf43f5-2f12-4b44-9eb4-e7a18eb6015f" />



3. Silhouette Plot
The silhouette plot provides a visual evaluation of each cluster. The plot shows that most points have a high silhouette score, confirming that the clusters are well-formed and distinct.

<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/218e349d-681c-4e63-a325-af681cb998b5" />
