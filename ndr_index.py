from scipy.spatial import distance
import numpy as np

"""
NDRindex (Normalization and Dimensionality Reduction index)
The function NDRindex takes as input a preprocessed data.
The NDRindex compute a score for the preprocessed data. 
This score should indicate the quality of the preprocessed data.
"""


def NDRindex(data):

    # Let n and d be the shape of the data.
    n, d = data.shape

    # Random index of a point
    A = np.random.choice(range(n))

    # Let K be equal to the index of a cluster.
    K = 1
    Y = np.full(n, -1)
    Y[A] = K

    # Geometric centers of the clusters.
    geometric_centers = {K: data[A]}

    # Calculate the average scale of the dataset.
    distances = distance.pdist(data, 'euclidean')
    M = np.percentile(distances, 25)
    average_scale = M / np.log10(n)

    # Stop when all data points are assigned to a cluster.
    while np.any(Y == -1):
        B_index = np.argmax(Y == -1)
        B = data[B_index]

        # Search the data point B that is closest to the geometric centers of cluster K.
        for j in range(n):
            point = data[j]
            if Y[j] == -1 and distance.euclidean(geometric_centers[K], point) < distance.euclidean(geometric_centers[K], B):
                B = point
                B_index = j

        # if the distance between the geometric center of K and the data point B is smaller than the average scale,
        # then add the data point to the cluster and update the geometric center
        # else let B a cluster on its own.
        if distance.euclidean(geometric_centers[K], B) < average_scale:
            Y[B_index] = K
            geometric_centers[K] = np.mean(data[Y == K], axis=0)

        else:
            K += 1
            Y[B_index] = K
            geometric_centers[K] = B

    # Calculate the score of the NDRindex
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