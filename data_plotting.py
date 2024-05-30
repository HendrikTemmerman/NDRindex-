import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median
import pandas as pd

"""
The function plot_datasets will create boxplots of the ARI data of each cluster algorithm. 
In the boxplot, we are also going to indicate with a red dot the ARI value of the combination chosen by the NDR index. 
"""
def plot_datasets(datasets):
    cluster_algorithms = ['ARI-hclust', 'ARI-kmeans', 'ARI-spectral']

    for cluster_algorithm in cluster_algorithms:
        ARI_NDRindex = []
        ARI_data = []

        for data in datasets:
            ARI_cluster = data[cluster_algorithm]
            highest_NDRindex = data['NDRindex'].idxmax()
            NDR = data.at[highest_NDRindex, cluster_algorithm]
            ARI_NDRindex.append(NDR)
            ARI_data.append(ARI_cluster)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.boxplot(ARI_data)
        ax.set_xlabel('Datasets')
        ax.set_ylabel('ARI')
        ax.set_title(f'{cluster_algorithm}')
        ax.set_xticklabels([f'Dataset {i + 1}' for i in range(len(datasets))])

        for i in range(len(ARI_data)):
            ax.scatter(i + 1, ARI_NDRindex[i], color='red', zorder=3, label='ARI of NDRindex choose' if i == 0 else "")

        ax.legend( loc='upper left')
        plt.savefig(f'ari_boxplots/{cluster_algorithm}.png')


"""
For each dataset, the function plot_correlation_ARI_RNA will generate a scatterplot 
comparing the NDRindex values with the ARI values from the various clustering algorithms
"""
def plot_correlation_ARI_RNA(datasets):
    cluster_algorithms = ['ARI-hclust', 'ARI-kmeans', 'ARI-spectral']
    colors = ['red', 'blue', 'green']

    for i, data in enumerate(datasets):
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        for j, cluster_algorithm in enumerate(cluster_algorithms):
            ARIs = data[cluster_algorithm]
            NDR = data['NDRindex']
            ax.scatter(NDR, ARIs, color=colors[j], marker='o', label=cluster_algorithm, zorder=3)

        ax.set_xlabel('NDRindex')
        ax.set_ylabel('ARI')
        ax.set_title(f'Correlation of ARI with NDRindex - Dataset {i + 1}')
        ax.legend(loc='upper right')
        plt.savefig(f'correlation_ari_ndr/correlation_ndr_ari_dataset{i + 1}.png')


"""
The function plot_correlation_combination_NDR will create a barplot for each dataset, 
displaying the NDR index values for each combination."""
def plot_correlation_combination_NDR(datasets):
    for i, data in enumerate(datasets):
        combination = data['Combination']
        NDR = data['NDRindex']
        fig = plt.figure(figsize=(40, 23))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.bar(combination, NDR, color='red', zorder=3)
        ax.set_xlabel('combination')
        ax.set_ylabel('NDrindex')
        ax.set_title(f'Dataset{i + 1}')
        plt.savefig(f'correlation_combination_ndr/Dataset {i + 1}.png')


"""
The function plot_combined_datasets_by_algorithm will create boxplots of the ARI data of each cluster algorithm. 
In the boxplot, we are also going to indicate with a red dot the ARI values of the combination chosen by the NDR index. 
The three boxplot figures are placed next to each other.
"""
def plot_combined_datasets_by_algorithm(datasets):
    cluster_algorithms = ['ARI-hclust', 'ARI-kmeans', 'ARI-spectral']
    dataset_labels = [f'Dataset {i + 1}' for i in range(len(datasets))]

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    for i, (data, ax) in enumerate(zip(datasets, axs)):
        ARI_NDRindex = []
        ARI_data = []

        for cluster_algorithm in cluster_algorithms:
            ARI_cluster = data[cluster_algorithm]
            highest_NDRindex = data['NDRindex'].idxmax()
            NDR = data.at[highest_NDRindex, cluster_algorithm]
            ARI_NDRindex.append(NDR)
            ARI_data.append(ARI_cluster)

        ax.boxplot(ARI_data)
        ax.set_xlabel('Clustering Algorithms')
        ax.set_ylabel('ARI')
        ax.set_title(f'Results for {dataset_labels[i]}')
        ax.set_xticklabels(cluster_algorithms)
        for j in range(len(ARI_data)):
            ax.scatter(j + 1, ARI_NDRindex[j], color='red', zorder=3, label='ARI of NDRindex choose' if j == 0 else "")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=1)

    plt.tight_layout()
    #Save the plots
    plt.savefig('ari_boxplots/combined_datasets.png')


"""Read the datasets"""
data1 = pd.read_csv('output_dataframes/data_1.csv')
data2 = pd.read_csv('output_dataframes/data_2.csv')
data3 = pd.read_csv('output_dataframes/data_3.csv')
datasets = [data1, data2, data3]

plot_combined_datasets_by_algorithm(datasets)
plot_datasets(datasets)
plot_correlation_ARI_RNA(datasets)
plot_correlation_combination_NDR(datasets)