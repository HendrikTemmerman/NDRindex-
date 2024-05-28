import copy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from ndr_index import NDRindex


"""Function to simulate data"""
def simulate_data(centers, cluster_std, n_samples, random_state):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std,
                      random_state=random_state)
    return X, y


"""Parameters for the datasets"""
centers = [[1, 1], [1, 9], [9, 1], [9, 9]]
cluster_std = [0.15, 0.5, 0.9, 1.5]
n_samples = 500
random_state = 42

"""Create three different datasets"""
simulated_datasets = [simulate_data(centers, std, n_samples, random_state) for std in cluster_std]

"""Plotting the simulated datasets"""
fig, axs = plt.subplots(1, 4, figsize=(20, 6))

"""Create scatter plots for each simulated dataset"""
for i, (X, y) in enumerate(simulated_datasets):
    axs[i].scatter(X[:, 0], X[:, 1], c=y)
    axs[i].set_title(f'Simulated Dataset {i + 1}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('y')

"""Display the simulated datasets"""
plt.tight_layout()
plt.show()

n = 51
ndr_values = np.zeros(len(simulated_datasets))
for i in range(n):
    print(f"Run {i + 1}")
    for sd in range(len(simulated_datasets)):
        dataset = simulated_datasets[sd]
        c = copy.deepcopy(dataset)
        ndr_index = NDRindex(c[0])
        ndr_values[sd] += ndr_index

ndr_values = ndr_values/n

"""Names of the datasets"""
datasets = ['Simulated Dataset 1', 'Simulated Dataset 2', 'Simulated Dataset 3', 'Simulated Dataset 4']

"""Plot the datasets"""
plt.figure(figsize=(10, 6))
plt.plot(datasets, ndr_values, marker='o', linestyle='-', color='b', markersize=8, linewidth=2)

"""Add titles and labels"""
plt.title('NDRindex Evolution of Simulated Datasets', fontsize=16, fontweight='bold')
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('NDRindex', fontsize=14)

"""Add grid for better readability"""
plt.grid(True, linestyle='--', alpha=0.7)

"""Customize tick parameters for better readability"""
plt.xticks(fontsize=12, rotation=15)
plt.yticks(fontsize=12)

"""Annotate values next to the points"""
for i, value in enumerate(ndr_values):
    plt.annotate(f'{value:.2f}', (datasets[i], ndr_values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

"""Display the plot"""
plt.tight_layout()
plt.show()