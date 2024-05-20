import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median
import pandas as pd


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


def plot_correlation_ARI_RNA(datasets):
    cluster_algorithms = ['ARI-hclust', 'ARI-kmeans', 'ARI-spectral']
    colors = ['red', 'blue', 'green']  # Different colors for each algorithm

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

data1 = pd.read_csv('output_dataframes/data_1.csv')
data2 = pd.read_csv('output_dataframes/data_2.csv')
data3 = pd.read_csv('output_dataframes/data_3.csv')
datasets = [data1, data2, data3]
plot_datasets(datasets)
plot_correlation_ARI_RNA(datasets)
plot_correlation_combination_NDR(datasets)