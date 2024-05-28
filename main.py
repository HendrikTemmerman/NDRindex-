import scanpy as sc
import copy as cp
import numpy as np
import pandas as pd
from load_data import datasets
from normalisation import normalizations
from dimensionality_reduction import dimension_reductions
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
from ndr_index import NDRindex #, SC3


LOAD_DATA = False
data_from_all_datasets = []

"""
The function clustering will cluster the preprocessed data (ndr_input) by using three clustering algorithms.
After that, we calculate the ARI of the clusters and add them to the data (dataframe).
"""


def clustering(data, ndr_input, true_labels, n_cell_types, combination, state):

    # The Agglomerative clustering algorithm.
    hclust = AgglomerativeClustering(n_clusters=n_cell_types, linkage='ward')
    hclust_labels = hclust.fit_predict(ndr_input)

    # The K-Means clustering algorithm.
    kmeans = KMeans(n_clusters=n_cell_types, random_state=state)
    kmeans_labels = kmeans.fit_predict(ndr_input)

    # The Spectral clustering algorithm.
    spectral = SpectralClustering(n_clusters=n_cell_types, affinity='nearest_neighbors', random_state=state)
    spectral_labels = spectral.fit_predict(ndr_input)

    # Calculate the ARI score and add it to the data.
    data['Combination'].append(combination)
    data['NDRindex'].append(NDRindex(ndr_input))
    data['ARI-hclust'].append(adjusted_rand_score(true_labels, hclust_labels))
    data['ARI-kmeans'].append(adjusted_rand_score(true_labels, kmeans_labels))
    data['ARI-spectral'].append(adjusted_rand_score(true_labels, spectral_labels))


"""
The function pre_process_data will pre process the data(dataset) with the normalization and the dimension reduction method.
"""
def pre_process_data(dataset, normalize, reduce_dimension, dataset_counter, combination, run):
    adata = cp.deepcopy(dataset)

    # Use the normalization method on the data.
    adata = normalize(adata)
    sc.pp.neighbors(adata, use_rep='X')

    # Use the dimension reduction method on the data.
    reduce_dimension(adata)

    # The raw data matrix (genes x cells).
    data_matrix = adata.X

    # If the data matrix is sparse, convert it to a dense format.
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

    # Create a dataframe of the ndr_input and save the dataframe to a csv file.
    df = pd.DataFrame(ndr_input)
    df.to_csv(f'preprocessed/{str(dataset_counter) + "+" + combination + "- " + str(run)}.csv', index=False)
    return ndr_input


"""
The function pipeline will calculate the NDRindex and the ARI of each combination for each normalization and dimension reduction method.
This is done by calling the function pre_process_data. We will do this an number_of_times. 
After that we will calculate the ARI of each clusters algorithm and save the values, this is done by calling the function clustering. 
"""


def pipeline(number_of_times):
    dataset_counter = 1

    # We run the experiment number_of_times for each dataset
    for dataset, true_labels, n_cell_types in datasets:
        df_list = []

        for run in range(1, number_of_times):
            data = {'Combination': [], 'NDRindex': [], 'ARI-hclust': [], 'ARI-kmeans': [], 'ARI-spectral': []}
            #data = {'Combination': [], 'SC3-ARI': []}

            # With each normalization and dimension reduction method we will pre process the data.
            # We calculate the NDRindex and the ARI of each combination for each normalization and dimension reduction method.
            # This will be done by calling the function pre_process_data and clustering
            for normalize in normalizations:
                for reduce_dimension in dimension_reductions:
                    combination = str(normalize.__name__) + "+" + str(reduce_dimension.__name__)
                    if LOAD_DATA:
                        df = pd.read_csv(f'preprocessed/{str(dataset_counter) + "+" + combination}.csv')
                        ndr_input = df.to_numpy()
                    else:
                        adata = cp.deepcopy(dataset)
                        adata = normalize(adata)
                        sc.pp.neighbors(adata, use_rep='X')
                        reduce_dimension(adata)
                        ndr_input = pre_process_data(dataset, normalize, reduce_dimension, dataset_counter, combination, run)

                    clustering(data, ndr_input, true_labels, n_cell_types, combination, run + dataset_counter)
                    #data['Combination'].append(combination)
                    #data['SC3-ARI'].append(SC3(adata, true_labels, n_cell_types))

            df_run = pd.DataFrame(data)
            df_list.append(df_run)

        df_combined = pd.concat(df_list)
        df_total = df_combined.groupby('Combination').mean().reset_index()

        data_from_all_datasets.append(df_total)
        # Save the calculations
        df_total.to_csv(f'output_sc3/data_{dataset_counter}.csv', index=False)
        dataset_counter += 1


pipeline(51)
