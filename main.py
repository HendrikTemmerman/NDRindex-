import scanpy as sc
import copy as cp
import numpy as np
import pandas as pd

#from load_data import datasets
from normalisation import normalizations
from dimensionality_reduction import dimension_reductions
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
from ndr_index import NDRindex
import matplotlib.pyplot as plt
from statistics import mean, median
from data_plotting import plot_correlation_ARI_RNA

LOAD_DATA = True
data_from_all_datasets = []


def clustering(data, ndr_input, true_labels, n_cell_types, combination, state):
    # Agglomerative Clustering
    hclust = AgglomerativeClustering(n_clusters=n_cell_types, linkage='ward')
    cluster_labels = hclust.fit_predict(ndr_input)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_cell_types, random_state=state)
    kmeans_labels = kmeans.fit_predict(ndr_input)

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=n_cell_types, affinity='nearest_neighbors', random_state=state)
    spectral_labels = spectral.fit_predict(ndr_input)

    data['Combination'].append(combination)
    data['NDRindex'].append(NDRindex(ndr_input))
    data['ARI-hclust'].append(adjusted_rand_score(true_labels, cluster_labels))
    data['ARI-kmeans'].append(adjusted_rand_score(true_labels, kmeans_labels))
    data['ARI-spectral'].append(adjusted_rand_score(true_labels, spectral_labels))


def pre_process_data(dataset, normalize, reduce_dimension, dataset_counter, combination, run):
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

    dimension_reduction_method = reduce_dimension.__name__
    for col in adata.obsm.keys():
        if dimension_reduction_method in col:
            reduced_data_matrix = adata.obsm[col]
            ndr_input = reduced_data_matrix
        else:
            ndr_input = dense_matrix

    df = pd.DataFrame(ndr_input)
    df.to_csv(f'preprocessed/{str(dataset_counter) + "+" + combination + "- " + str(run)}.csv', index=False)

    return ndr_input


def pipeline(number_of_times):
    dataset_counter = 1
    for dataset, true_labels, n_cell_types in datasets:
        df_list = []
        for run in range(1, number_of_times):
            print(f"Run number {run} for dataset {dataset_counter}")
            data = {'Combination': [], 'NDRindex': [], 'ARI-hclust': [], 'ARI-kmeans': [], 'ARI-spectral': []}
            for normalize in normalizations:
                for reduce_dimension in dimension_reductions:
                    combination = str(normalize.__name__) + "+" + str(reduce_dimension.__name__)
                    if LOAD_DATA:
                        df = pd.read_csv(f'preprocessed/{str(dataset_counter) + "+" + combination}.csv')
                        ndr_input = df.to_numpy()
                    else:
                        ndr_input = pre_process_data(dataset, normalize, reduce_dimension, dataset_counter, combination,
                                                     run)

                    clustering(data, ndr_input, true_labels, n_cell_types, combination, run + dataset_counter)

            df_run = pd.DataFrame(data)
            df_list.append(df_run)

        df_combined = pd.concat(df_list)
        df_total = df_combined.groupby('Combination').mean().reset_index()

        data_from_all_datasets.append(df_total)
        df_total.to_csv(f'output_dataframes/data_{dataset_counter}.csv', index=False)
        dataset_counter += 1


# pipeline(51)
