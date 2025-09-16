🐧 Penguins Clustering with KMeans

This project applies unsupervised learning techniques to cluster penguin species based on the Palmer Penguins dataset
. Using KMeans clustering, feature engineering, and evaluation metrics, we analyze how well the algorithm can separate species without labels.

📌 Project Overview

Preprocessed the dataset by dropping irrelevant columns, encoding categorical variables, and normalizing features.

Applied KMeans clustering with the Elbow Method to determine the optimal number of clusters.

Evaluated clustering quality using Purity and Silhouette Score.

Visualized results with scatterplots, centroid plots, and silhouette diagrams.

Compared learned clusters to true species labels to assess model effectiveness.

⚙️ Tech Stack

Python

Pandas, NumPy, Matplotlib, Seaborn → data preprocessing & visualization

Scikit-learn → KMeans, scaling, clustering metrics

SciPy → distance calculations

📊 Key Results

Optimal number of clusters: 3 (matching the true number of penguin species).
<img width="220" height="78" alt="image" src="https://github.com/user-attachments/assets/7d5642c0-0356-4a06-8089-a77a8b0f6978" />

<img width="473" height="75" alt="image" src="https://github.com/user-attachments/assets/00763966-b92b-453b-b5a1-71291595a181" />


Purity: ~0.9, showing strong alignment between clusters and species.
<img width="128" height="22" alt="image" src="https://github.com/user-attachments/assets/8a0fec9f-cc30-4829-a695-bcf737c577f1" />

Silhouette Score: Demonstrated good cluster separation and compactness.
<img width="249" height="28" alt="image" src="https://github.com/user-attachments/assets/f7a8abc5-6caa-45e0-9362-6ba0afe850ff" />


📈 Visualizations

Elbow Method to choose k.
<img width="398" height="221" alt="image" src="https://github.com/user-attachments/assets/df49e6f3-fb19-4861-a1c9-47c5a449e7f5" />


Cluster plots with centroids highlighted.
<img width="422" height="272" alt="image" src="https://github.com/user-attachments/assets/37cc879f-9fff-4d32-be6f-7c9c14c92731" />


Silhouette plot to evaluate intra-cluster cohesion vs inter-cluster separation.
<img width="430" height="275" alt="image" src="https://github.com/user-attachments/assets/311ac004-518a-4dea-aede-99215490683f" />
