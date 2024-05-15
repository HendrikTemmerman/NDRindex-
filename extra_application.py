import pandas as pd
import numpy as np
import scanpy as sc
from scipy.io import arff
from anndata import AnnData
from normalisation import normalizations
from dimensionality_reduction import dimension_reductions
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from ndr_index import NDRindex

import matplotlib.pyplot as plt
import seaborn as sns

arff_file = arff.loadarff('phpSSK7iA.arff')


df = pd.DataFrame(arff_file[0])

df['target'].replace(b'1', 1, inplace=True)
df['target'].replace(b'0', 0, inplace=True)

true_labels = df['target']

n_responses = 2

df = df.drop(columns=['target'], axis=1)

adata = AnnData(X=df)

data = {'Combination': [],
        'NDRindex': [],
        'ARI-hclust': [],
        'ARI-kmeans': [],
        'ARI-dbscan': [],
        'ARI-spectral': []}

for normalize in normalizations:
    for reduce_dimension in dimension_reductions:
        combination = str(normalize.__name__) + " + " + str(reduce_dimension.__name__)
        print(combination)

        adata = normalize(adata)
        sc.pp.neighbors(adata, use_rep='X')
        reduce_dimension(adata)

        # If you just want the raw data matrix (genes x cells)
        data_matrix = adata.X

        # If the data matrix is sparse, convert it to a dense format
        if isinstance(data_matrix, np.ndarray):
            dense_matrix = data_matrix
        else:
            dense_matrix = data_matrix.toarray()

        dimension_reduction_method = reduce_dimension.__name__
        for col in adata.obsm.keys():
            if dimension_reduction_method in col:
                reduced_data_matrix = adata.obsm[col]
                ndr_input = reduced_data_matrix
            else:
                ndr_input = dense_matrix

        # Agglomerative Clustering
        hclust = AgglomerativeClustering(n_clusters=n_responses, linkage='ward')
        cluster_labels = hclust.fit_predict(ndr_input)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_responses, random_state=42)
        kmeans_labels = kmeans.fit_predict(ndr_input)


        # Spectral Clustering
        spectral = SpectralClustering(n_clusters=n_responses, affinity='nearest_neighbors',
                                      random_state=42)
        spectral_labels = spectral.fit_predict(ndr_input)

        data['Combination'].append(combination)
        data['NDRindex'].append(NDRindex(ndr_input))
        data['ARI-hclust'].append(adjusted_rand_score(true_labels, cluster_labels))
        data['ARI-kmeans'].append(adjusted_rand_score(true_labels, kmeans_labels))
        data['ARI-dbscan'].append(adjusted_rand_score(true_labels, dbscan_labels))
        data['ARI-spectral'].append(adjusted_rand_score(true_labels, spectral_labels))

df = pd.DataFrame(data)
print(df)

df['Dataset'] = df['combination'].apply(lambda x: x.split('+')[0].strip())

# Set up the matplotlib figure and axes
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)  # Adjust figsize as needed
fig.suptitle('Clustering Algorithm Performance Comparison')

# List of clustering methods
clustering_methods = ['ARI-hclust', 'ARI-kmeans', 'ARI-dbscan', 'ARI-spectral']
titles = ['Hierarchical Clustering', 'K-Means', 'DBSCAN', 'Spectral Clustering']

# Plotting
for ax, method, title in zip(axes, clustering_methods, titles):
    # Subset dataframe for the method
    sns.barplot(data=df, x='Dataset', y=method, hue='combination', ax=ax, palette='viridis')
    ax.set_title(title)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.legend(title='Normalization', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title and legend
plt.show()





