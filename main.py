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
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from statistics import mean, median



def plot_data(data, NDRindex, name):

    left = [1, 2, 3, 4, 5]

    ARI_NDRindex = NDRindex
    ARI_average = mean(data)
    ARI_median = median(data)
    ARI_upper_quantile = np.quantile(data, 0.75)
    ARI_max = max(data)

    height = [ARI_NDRindex, ARI_average, ARI_median, ARI_upper_quantile, ARI_max]

    tick_label = ['ARI of NDRindex', 'ARI average', 'ARI median', 'ARI upper quantile', 'ARI max']

    plt.figure(figsize=(10, 6))
    plt.bar(left, height, tick_label=tick_label,width=0.8, color=['red', 'green', 'blue', 'orange', 'purple'])
    plt.ylabel('ARI')
    plt.title(name)
    plt.savefig(f'{name}.png')
    plt.show()





counter = 1



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
            #data['ARI-Ap_clust'].append(adjusted_rand_score(true_labels, af_labels))


    df = pd.DataFrame(data)
    df.to_csv(f"data_{counter}/data")
    highest_NDRindex = df['NDRindex'].idxmax()
    print("the highest NDR index",highest_NDRindex)
    plot_data(data['ARI-hclust'], df.at[highest_NDRindex, 'ARI-hclust'],f"data_{counter}/hclust")
    plot_data(data['ARI-kmeans'], df.at[highest_NDRindex, 'ARI-kmeans'],f"data_{counter}/kmeans")
    plot_data(data['ARI-dbscan'], df.at[highest_NDRindex, 'ARI-dbscan'],f"data_{counter}/dbscan")
    plot_data(data['ARI-spectral'], df.at[highest_NDRindex, 'ARI-spectral'],f"data_{counter}/spectral")
    counter = counter + 1
    print(df)
