import copy

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance
from numpy.random import choice
import numpy as np
from sklearn.metrics import adjusted_rand_score
from NDRindex import NDRindex


# Function to simulate data similar to the provided image
def simulate_data(centers, cluster_std, n_samples, random_state):
    """
    Simulates clustering data with make_blobs.

    Parameters:
        centers (int): Number of centers to generate.
        cluster_std (list of floats): The standard deviation of the clusters.
        n_samples (int): The total number of points equally divided among clusters.
        random_state (int): Determines random number generation for dataset creation.

    Returns:
        X (array of [n_samples, 2]): The generated samples.
        y (array of [n_samples]): The integer labels for cluster membership of each sample.
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std,
                      random_state=random_state)
    return X, y


# Parameters for the datasets
centers = [[1, 1], [1, 9], [9, 1], [9, 9]]
cluster_std = [0.15, 0.5, 0.75, 1.5]
n_samples = 500
random_state = 42

# Create three different datasets
simulated_datasets = [simulate_data(centers, std, n_samples, random_state) for std in cluster_std]

# Plotting the simulated datasets
fig, axs = plt.subplots(1, 4, figsize=(20, 6))

# Display the datasets


def NDRindexSimulation(X):


    n, d = X.shape
    distances = distance.pdist(X, 'euclidean')
    M = np.percentile(distances, 25)


    average_scale = M / np.log10(n)
    # Initialize clusters
    gcenter = {}
    Y = np.full(n, -1)  # Cluster assignments

    K = 1
    R = 0
    A = np.random.choice(range(n))
    Y[A] = K
    gcenter[K] = X[A]

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
    NDRindex = 1.0 - R / average_scale
    print(NDRindex)

    #print("ARI:", adjusted_rand_score(y, Y))
    return NDRindex



for i, (X, y) in enumerate(simulated_datasets):
    axs[i].scatter(X[:, 0], X[:, 1], c=y)
    axs[i].set_title(f'Simulated Dataset {i + 1}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

plt.tight_layout()
plt.show()

for sd in simulated_datasets:
    print("---------------")
    c = copy.deepcopy(sd)
    NDRindex(c[0])
    print("and")
    NDRindexSimulation(sd[0])