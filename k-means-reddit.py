# k-means-reddit.py            By: Samer Al-khateeb
# To run this script, you need to install the needed librarires by typing:
# pip install pandas scikit-learn matplotlib seaborn numpy kneed

# importing the needed libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from kneed import KneeLocator #pip install kneed ...to find the cluster in the elbow if you can't tell from the graph


# Specify the columns you want to load
columns_to_load = ['postNumComments', 'postNumCrossPosts', 
                   'postUpvoteRatio', 'postUps', 
                   'postDowns', 'postScore']

# 0) Import the data from CSV file and convert it into Pandas DataFrame
# Read the CSV file with selected columns
data = pd.read_csv('reddit_posts_output.csv', usecols=columns_to_load) # Replace with your actual file path

print(data)

# 1) Preprocessing the Data:
# Identify and encode categorical columns
#label_encoder = LabelEncoder()
#for column in data.select_dtypes(include=['object']).columns:
#    data[column] = label_encoder.fit_transform(data[column])

# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Impute missing values using the mean or median
#imputer = SimpleImputer(strategy='median') #median
#X = imputer.fit_transform(numeric_data)

# if you do not want to impute/estimate missing values, you can remove the rows with missing values
#X = data.dropna()  # Drops any row that has at least one NaN value
# OR fill missing values with Zeros
X = data.fillna(0)  # Replaces all NaN values with 0
#save the cleaned data into a csv file
X.to_csv('cleaned_reddit_data.csv', index=False)  # Saves without the index column

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

'''

# 2) Determine Optimal PCA Components (if data dimensions in hundreds or more, we should reduce the dimensionality)
pca = PCA() #Principal Component Analysis (PCA)
pca.fit(data_scaled)
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.show()

# Choose n_components where cumulative variance > 95%
n_components = np.argmax(explained_variance >= 0.95) + 1
print(f"Optimal number of components: {n_components}")
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data_scaled)
'''

# 3) Applying K-Means multiple times to Find Optimal K
# The two plots should show the optimal K...
# the Distortion or Inertia plot should show the optimal K, i.e., where the elbow is. It identifies 
# the "elbow" point, where the rate of decrease slows down. The optimal K is typically the point where the plot bends, 
# meaning adding more clusters doesn’t significantly improve compactness.
# The silhouette scores plot should show the optimal K, i.e., the cluster with the highest silhouette score
inertia_values = []
silhouette_scores = []

# range of K to try...
K_range = range(2, 30)
# applying k means and appending results
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_scaled) # data_pca
    score = silhouette_score(data_scaled, clusters) # data_pca
    silhouette_scores.append(score)
    inertia_values.append(kmeans.inertia_)

# Find the optimal K using KneeLocator and inertia_values
knee_locator = KneeLocator(K_range, inertia_values, curve="convex", direction="decreasing")
optimal_k = knee_locator.knee
if optimal_k is None:
    print("Warning: No clear elbow detected. Consider looking at the graph manually.")
else:
    print(f"Optimal number of clusters (K) based on the elbow method: {optimal_k}")
# Plot Inertia vs Number of Clusters
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia_values, marker='o', linestyle='-', color='b', label='Inertia')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K: {optimal_k}')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Distortion (Inertia: Sum of Squared Distances)')
plt.title('Elbow Method: Inertia vs. Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()

# Find the optimal K using  silhouette_scores, need to pick the maximum
#optimal_k = max(zip(K_range, silhouette_scores), key=lambda x: x[1])[0]
if optimal_k is None:
    print("Warning: No clear elbow detected. Consider looking at the graph manually.")
else:
    print(f"Optimal number of clusters (K) based on the silhouette scores is: {optimal_k}")

# Plot Silhouette Score vs Number of Clusters
'''
The silhouette score ranges from -1 to 1:
1.0 → Perfect clustering, with well-separated clusters.
0.5+ → Good clustering with clear separations.
0.2 - 0.5 → Weak clustering, indicating overlapping clusters.
< 0.2 → Poor clustering; clusters are not well defined.
'''
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color='b', label='Silhouette Score')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K: {optimal_k}')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()





#############BY NOW YOU SHOULD KNOW THE OPTIMAL NUMBER OF CLUSTERS######################
### ADJUST THE CODE BELOW ###
# 4) Running K-Means - Adjust optimal_k based on the plots above
optimal_k = XX # Update XX based on the plots
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_scaled) # data_pca

#5) Evaluating Results using: 
print(f'Using the optimal number of clusters: {optimal_k}')
# Silhouette Score on Scaled Data, higher values are better.
score = silhouette_score(data_scaled, clusters) # data_pca
print(f'The Silhouette Score is: {score:.2f}')
# Inertia (within-cluster sum of squares) lower values are better (indicating compact clusters).
print(f'The Inertia (Sum of Squared Distances) is: {kmeans.inertia_:.2f}')


#6) Convert NumPy array to DataFrame
df_scaled = pd.DataFrame(data_scaled, columns=[f'Feature_{i}' for i in range(data_scaled.shape[1])])
# Add the original index column
df_scaled['Original_Index'] = df_scaled.index
# Add cluster labels
df_scaled['Cluster Assignment'] = clusters
# Save to CSV, including the original indices
df_scaled.to_csv('reddit_clustered_data.csv', index=False)
print("Cluster results with original indices saved to 'clustered_data_with_index.csv'")


'''
# 7) Visualizing with t-distributed Stochastic Neighbor Embedding (t-SNE)
# --- Experimenting with different perplexity values here ---
Higher perplexity (e.g., 30 to 50), t-SNE tries to consider more distant points as "neighbors" (smooths out and includes broader groups)
lower perplexity (e.g., 5 to 10) means it focuses more tightly on very close points (emphasizes fine local structures)
Typical values: 5 to 50, but need to tune based on your data.
For the visualization affect, think of perplexity like asking:
For each point, how many neighbors should I assume it has?
'''
perplexities = [5, 10, 30, 40]  # Example perplexity values to try
for perplexity in perplexities:
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    data_tsne = tsne.fit_transform(data_scaled)  # data_pca
    
    # Plotting the t-SNE visualization
    df_plot = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2'])
    df_plot['Cluster'] = clusters
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', palette='bright', data=df_plot)
    plt.title(f't-SNE with Perplexity={perplexity}')
    plt.show()
