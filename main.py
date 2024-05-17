import scanpy as sc
import copy as cp
import numpy as np
import pandas as pd

from load_data import datasets
from normalisation import normalizations
from dimensionality_reduction import dimension_reductions
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
from ndr_index import NDRindex
import matplotlib.pyplot as plt
from statistics import mean, median
from data_plotting import plot_datasets

LOAD_DATA = True
data_from_all_datasets = []


def clustering(data, ndr_input, true_labels, n_cell_types, combination):
    # Agglomerative Clustering
    hclust = AgglomerativeClustering(n_clusters=n_cell_types, linkage='ward')
    cluster_labels = hclust.fit_predict(ndr_input)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_cell_types, random_state=42)
    kmeans_labels = kmeans.fit_predict(ndr_input)

    # Affinity Propagation
    af = AffinityPropagation(random_state=0, max_iter=1000, damping=0.99)
    af_labels = af.fit_predict(ndr_input)

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_cell_types, affinity='nearest_neighbors', random_state=42)
    spectral_labels = spectral.fit_predict(ndr_input)

    data['Combination'].append(combination)
    data['NDRindex'].append(NDRindex(ndr_input))
    data['ARI-hclust'].append(adjusted_rand_score(true_labels, cluster_labels))
    data['ARI-kmeans'].append(adjusted_rand_score(true_labels, kmeans_labels))
    data['ARI-spectral'].append(adjusted_rand_score(true_labels, spectral_labels))
    data['ARI-ap_clust'].append(adjusted_rand_score(true_labels, af_labels))



def pipeline(number_of_times):
    for time in range(number_of_times):
        dataset_counter = 1
        for dataset, true_labels, n_cell_types in datasets:
            data = {'Combination': [],
                    'NDRindex': [],
                    'ARI-hclust': [],
                    'ARI-kmeans': [],
                    'ARI-spectral': [],
                    'ARI-ap_clust': []}
            for normalize in normalizations:
                for reduce_dimension in dimension_reductions:
                    print("--------------------------------")
                    combination = str(normalize.__name__) + "+" + str(reduce_dimension.__name__)
                    print(combination)
                    if LOAD_DATA == True:
                        print("it's true")
                        df = pd.read_csv(f'preprocessed/{str(dataset_counter) + "+" + combination}.csv')
                        ndr_input = df.to_numpy()
                    else:
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

                        df = pd.DataFrame(ndr_input)
                        df.to_csv(f'test_preprocessed/{str(dataset_counter) + "+" + combination + "- " + str(time)}.csv', index=False)

                    clustering(data, ndr_input, true_labels, n_cell_types, combination)

            df = pd.DataFrame(data)
            data_from_all_datasets.append(df)
            df.to_csv(f'ari_data/data_{dataset_counter}.csv', index=False)
            dataset_counter += 1

pipeline(1)
#plot_datasets(data_from_all_datasets)
