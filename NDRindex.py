from scipy.spatial import distance
from numpy.random import choice
import numpy as np
from sklearn.metrics import adjusted_rand_score


def NDRindex(data, true_y):

    X = data
    n = data.shape[0]

    M = np.median([distance.euclidean(X[i], X[j]) for i in range(n) for j in range(i + 1, n)])


    average_scale = M / np.log10(n)
    # Initialize clusters
    gcenter = {}
    Y = np.full(n, -1)  # Cluster assignments
    print(Y)

    K = 1
    R = 0
    random_point = choice(range(n))
    Y[random_point] = K
    gcenter[K] = X[random_point]

    while np.any(Y == -1):  # While there are points not assigned to a cluster
        for j in range(n):
            if Y[j] == -1:  # If point j is not assigned to a cluster
                distances_to_centers = [distance.euclidean(gcenter[k], X[j]) for k in gcenter]
                nearest_cluster = np.argmin(distances_to_centers) + 1
                if distances_to_centers[nearest_cluster - 1] < average_scale:
                    Y[j] = nearest_cluster
                    cluster_points = X[Y == nearest_cluster]
                    gcenter[nearest_cluster] = np.mean(cluster_points, axis=0)
                else:
                    K += 1
                    Y[j] = K
                    gcenter[K] = X[j]

    # Calculate NDRindex
    for k in gcenter:
        cluster_points = X[Y == k]
        R += sum([distance.euclidean(gcenter[k], p) for p in cluster_points]) / len(cluster_points)


    R = R / K
    ndrindex = 1.0 - R / average_scale
    print("NDRindex:", ndrindex)
    print(Y[0:10])
    print("ARI:", adjusted_rand_score(true_y, Y))
    return ndrindex
