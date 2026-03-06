# Penguin Species Clustering with KMeans

An unsupervised machine learning project that clusters penguin species from the [Palmer Penguins LTER dataset](https://pallter.marine.rutgers.edu/) using KMeans. The goal is to evaluate how well an unsupervised algorithm can recover the true species labels (Adelie, Chinstrap, Gentoo) without ever seeing them during training.

## Table of Contents

- [Dataset](#dataset)
- [Approach and Design Choices](#approach-and-design-choices)
- [Results](#results)
- [Visualizations](#visualizations)
- [Tech Stack](#tech-stack)

## Dataset

The dataset comes from the Palmer Long-Term Ecological Research (LTER) program and contains 344 observations of penguins from three species collected across three islands in the Palmer Archipelago, Antarctica. Each record includes morphological measurements (culmen length and depth, flipper length, body mass), isotope ratios (Delta 15 N, Delta 13 C), and metadata (island, sex, study name, etc.).

The choice of this dataset is motivated by its well-separated species structure, which makes it a good benchmark for clustering: we know there are 3 species and can measure how accurately KMeans recovers them.

## Approach and Design Choices

### Why KMeans?

KMeans was chosen because the problem has a clear expected structure (3 compact, roughly spherical groups of species). KMeans is well-suited for this kind of globular cluster geometry and is computationally efficient, making it a natural first choice before considering more complex algorithms like DBSCAN or Gaussian Mixture Models.

### Feature Selection and Encoding

Irrelevant metadata columns (`studyName`, `Sample Number`, `Individual ID`, `Date Egg`, `Comments`) are dropped since they carry no biological signal. The remaining categorical variables (`Region`, `Island`, `Stage`) are one-hot encoded to convert them into numeric features that KMeans can process. `drop_first=True` is used to avoid multicollinearity between the dummy variables.

Columns like `Sex` and `Clutch Completion` are excluded from the feature set because they do not discriminate between species — both male and female penguins exist across all three species, and clutch completion is a reproductive status, not a morphological trait.

This encoding strategy allows the model to leverage categorical context (especially island information, since species distribution varies by island) alongside the continuous morphological measurements (culmen length, culmen depth, flipper length, body mass) and isotope ratios.

### Why StandardScaler Normalization?

All features are standardized to zero mean and unit variance using StandardScaler (or an equivalent manual implementation). This is essential for KMeans because the algorithm relies on Euclidean distance — without normalization, features with larger scales (e.g., body mass in grams vs. culmen length in millimeters) would dominate the distance computation and bias cluster assignments.

### Handling Missing Values

Rows with missing values are dropped rather than imputed. The dataset has relatively few missing entries, so dropping them preserves data integrity without introducing artificial values that could distort the cluster structure. Imputation (e.g., mean or KNN-based) would be preferable for larger amounts of missing data, but here simplicity is favored.

### Choosing k with the Elbow Method

The optimal number of clusters is determined using the Elbow Method, which plots inertia (within-cluster sum of squared distances) against increasing values of k. The "elbow" — the point where adding more clusters yields diminishing returns — is found at **k = 3**, which matches the known number of species in the dataset.

### Why Scikit-learn?

The project uses scikit-learn's `KMeans` implementation, which provides a well-tested, optimized version of Lloyd's algorithm with multiple random initializations (`n_init=10`) to avoid poor local minima. Scikit-learn is also used for `StandardScaler` and silhouette metrics, keeping the pipeline consistent and reliable.

## Results

### Optimal Number of Clusters: 3

The Elbow Method clearly identifies k = 3 as the optimal number of clusters, consistent with the three penguin species in the dataset.

![image](https://github.com/user-attachments/assets/4e2aee3e-739b-4801-9079-3447acde7bfc)
![image](https://github.com/user-attachments/assets/9f3f5d88-a21e-4be7-b8c8-7f5f321c9cb0)

### Purity Score: 0.92

Purity measures the fraction of data points in each cluster that belong to the majority class. A score of 0.92 indicates that 92% of points are correctly grouped with members of their own species, demonstrating strong alignment between the unsupervised clusters and the true labels.

![image](https://github.com/user-attachments/assets/cb70104e-229e-4aea-9b6b-c2f1b1069bdd)

### Silhouette Score

The Silhouette Score measures how similar each point is to its own cluster compared to the nearest neighboring cluster. Values range from -1 (wrong cluster) to +1 (well-matched). The average silhouette score confirms good cluster cohesion and separation.

![image](https://github.com/user-attachments/assets/e99cde90-1742-45a4-9a49-9601bdf64f8c)

## Visualizations

### Elbow Method

The sum of squared distances (inertia) decreases as k increases. The elbow at k = 3 shows that going beyond 3 clusters provides minimal improvement, confirming the natural structure of the data.

![Figure_1](https://github.com/user-attachments/assets/d9595dce-b370-405a-ba88-23b3fb2bba9e)

### Cluster Scatterplot with Centroids

Data points are projected onto two features (culmen length vs. culmen depth) and colored by cluster assignment. The black "X" markers indicate the cluster centroids. The clear spatial separation between clusters reflects the morphological differences between species.

![Figure_1](https://github.com/user-attachments/assets/cf844e68-431d-4bfb-8440-cfb34bd4c9f7)

### Silhouette Plot

Each horizontal bar represents a data point, ordered by silhouette value within its cluster. Most points have high positive values, meaning they are well-placed in their assigned cluster. The red dashed line shows the global average. Clusters with uniform, high silhouette values indicate well-defined, compact groups.

![Figure_1](https://github.com/user-attachments/assets/218e349d-681c-4e63-a325-af681cb998b5)

## Tech Stack

| Library | Purpose |
|---|---|
| **Pandas** | Data loading, manipulation, and crosstab analysis |
| **NumPy** | Numerical operations and array handling |
| **Matplotlib** | Plotting (elbow method, silhouette diagrams) |
| **Seaborn** | Cluster scatterplot visualizations |
| **Scikit-learn** | StandardScaler, KMeans, silhouette metrics |
| **SciPy** | Centroid distance matrix computation (`cdist`) |

