from scipy.spatial import distance
import anndata as ad
import scanpy as sc

import numpy as np
import pandas as pd
import sc3s
from sklearn.metrics import adjusted_rand_score

from load_data import datasets

count = 0


def NDRindex(data):
    n, d = data.shape
    A = np.random.choice(range(n))

    K = 1

    Y = np.full(n, -1)
    Y[A] = K
    geometric_centers = {K: data[A]}

    distances = distance.pdist(data, 'euclidean')
    M = np.percentile(distances, 25)

    average_scale = M / np.log10(n)

    while np.any(Y == -1):
        B_index = np.argmax(Y == -1)
        B = data[B_index]

        for j in range(n):
            point = data[j]
            if Y[j] == -1 and distance.euclidean(geometric_centers[K], point) < distance.euclidean(geometric_centers[K],
                                                                                                   B):
                B = point
                B_index = j

        if distance.euclidean(geometric_centers[K], B) < average_scale:
            Y[B_index] = K
            geometric_centers[K] = np.mean(data[Y == K], axis=0)

        else:
            K += 1
            Y[B_index] = K
            geometric_centers[K] = B

    R = 0
    for i in geometric_centers:
        points_i = data[Y == i]
        size_i = len(points_i)
        average_distance = 0
        for p in points_i:
            average_distance += distance.euclidean(p, geometric_centers[i])
        average_distance /= size_i
        R += average_distance
    R /= K

    NDRindex = 1.0 - (R / average_scale)
    return NDRindex


def SC3(adata, true_labels, n_cell_types):
    sc3s.tl.consensus(adata, n_clusters=n_cell_types)
    return adjusted_rand_score(true_labels, adata.obs[f'sc3s_{n_cell_types}'])
