import scanpy as sc
import copy as cp
import numpy as np
import pandas as pd

from Datasets import datasets
from Normalization import normalizations
from DimensionalityReduction import dimension_reductions
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from NDRindex import NDRindex

for dataset, true_labels, n_cell_types in datasets:
    data = {'Combination': [],
            'NDRindex': [],
            'ARI-hclust': [],
            'ARI-kmeans': [],
            'ARI-dbscan': [],
            'ARI-spectral': []}
    for normalize in normalizations:
        for reduce_dimension in dimension_reductions:
            print("--------------------------------")
            combination = str(normalize.__name__) + " + " + str(reduce_dimension.__name__)
            print(combination)

            adata = cp.deepcopy(dataset)
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

            # Now, if you have already performed some dimensionality reduction and it is stored in .obsm
            # For example, if you have PCA results in .obsm['X_pca']
            dimension_reduction_method = reduce_dimension.__name__
            for col in adata.obsm.keys():
                if dimension_reduction_method in col:
                    reduced_data_matrix = adata.obsm[col]
                    ndr_input = reduced_data_matrix
                else:
                    ndr_input = dense_matrix

                # Agglomerative Clustering
                hclust = AgglomerativeClustering(n_clusters=n_cell_types, linkage='ward')
                cluster_labels = hclust.fit_predict(ndr_input)

                # K-Means Clustering
                kmeans = KMeans(n_clusters=n_cell_types, random_state=42)
                kmeans_labels = kmeans.fit_predict(ndr_input)

                # DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(ndr_input)

                # Spectral Clustering
                spectral = SpectralClustering(n_clusters=n_cell_types, affinity='nearest_neighbors',
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
