import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
import scanpy as sc


def TMM(data):
    # Aannemende dat een functie om TMM normalisatie te doen beschikbaar is in bijv. `scanpy` of anders.
    import rpy2.robjects as ro
    ro.r('source("path_to_TMM_normalization_function.R")')
    norm_factors = ro.r('calcNormFactors')(data, method="TMM")
    sec = norm_factors
    tsec = np.where(sec > 0, 1, -1)
    sec = np.abs(sec)
    res = np.log2((data.T / sec).T + 1)
    res = (data.T / tsec).T
    return res


def scarn(data):
    # Vergelijkbaar, moet worden aangepast afhankelijk van de beschikbare functies.
    sec = ro.r('computeSumFactors')(data)
    tsec = np.where(sec > 0, 1, -1)
    sec = np.abs(sec)
    res = np.log2((data.T / sec).T + 1)
    res = (data.T / tsec).T
    return res


def pca(data):
    pca_model = PCA(n_components=2)
    res = pca_model.fit_transform(data.T)
    return res


def seurat(data):
    adata = sc.AnnData(data)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata.X


def NDRselect(data, norm_funcs, red_funcs, norm_names, red_names, cluster_name, cluster_number):
    results_list = []
    max_index = 0
    best_data = None
    for norm_func, norm_name in zip(norm_funcs, norm_names):
        normalized_data = norm_func(data)
        for red_func, red_name in zip(red_funcs, red_names):
            reduced_data = red_func(normalized_data)
            ndr_index = calculate_ndr_index(reduced_data)  # Dummy functie om NDRindex te berekenen
            results_list.append((ndr_index, norm_name, red_name))
            if ndr_index > max_index:
                max_index = ndr_index
                best_data = reduced_data

    results_list.sort(reverse=True, key=lambda x: x[0])

    clusters = None
    if cluster_name == 'kmeans':
        clusters = KMeans(n_clusters=cluster_number).fit_predict(best_data)
    elif cluster_name == 'hclust':
        hc = linkage(pdist(best_data), method='average')
        clusters = cut_tree(hc, n_clusters=cluster_number).flatten()

    return {
        'results_list': results_list,
        'best_data': best_data,
        'clusters': clusters
    }


# Voorbeeld van functies die als parameters kunnen worden gebruikt
norm_funcs = [TMM, scarn, seurat]
red_funcs = [pca]
norm_names = ['TMM', 'scarn', 'seurat']
red_names = ['pca']

# Aanroepen van de functie
result = NDRselect(your_data, norm_funcs, red_funcs, norm_names, red_names, 'kmeans', 4)
