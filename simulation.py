import copy

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance
from numpy.random import choice
import numpy as np
from sklearn.metrics import adjusted_rand_score
from ndr_index import NDRindex


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
cluster_std = [0.15, 0.5, 0.9, 1.5]
n_samples = 500
random_state = 42

# Create three different datasets
simulated_datasets = [simulate_data(centers, std, n_samples, random_state) for std in cluster_std]

# Plotting the simulated datasets
fig, axs = plt.subplots(1, 4, figsize=(20, 6))





for i, (X, y) in enumerate(simulated_datasets):
    axs[i].scatter(X[:, 0], X[:, 1], c=y)
    axs[i].set_title(f'Simulated Dataset {i + 1}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

plt.tight_layout()
plt.show()

n = 10
ndr_values = np.zeros(len(simulated_datasets))
for i in range(n):
    print(f"Run {i + 1}")
    for sd in range(len(simulated_datasets)):
        dataset = simulated_datasets[sd]
        c = copy.deepcopy(dataset)
        ndr_index = NDRindex(c[0])
        ndr_values[sd] += ndr_index

ndr_values = ndr_values/n

datasets = ['Simulated Dataset 1', 'Simulated Dataset 2', 'Simulated Dataset 3', 'Simulated Dataset 4']

# Plotting the NDRindex values
plt.figure(figsize=(10, 6))
plt.plot(datasets, ndr_values, marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=8)

# Adding titles and labels
plt.title('NDRindex Values for Different Datasets')
plt.xlabel('Datasets')
plt.ylabel('NDRindex Value')

# Adding the values on top of the points
for i, value in enumerate(ndr_values):
    plt.text(i, value + 0.01, str(value), ha='center', va='bottom')

# Display the plot
plt.ylim(0, 1)  # Set y-axis limit to better visualize the values
plt.grid(True)
plt.show()