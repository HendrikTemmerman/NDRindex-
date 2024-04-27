import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
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

for i, (X, y) in enumerate(simulated_datasets):
    axs[i].scatter(X[:, 0], X[:, 1], c=y)
    axs[i].set_title(f'Simulated Dataset {i + 1}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

plt.tight_layout()
plt.show()

for sd in simulated_datasets:
    NDRindex(sd)