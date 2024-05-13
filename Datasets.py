import scanpy as sc
import pandas as pd
import numpy as np
from Normalization import tmm


import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


"""
Dataset 1: MD704 (4 cell types)
Dataset 2: MD710 (3 cell types)
Dataset 3: MD711 (4 cell types)
"""

"""Dataset 1"""
dataset1 = sc.read_10x_mtx("dataset1")
metadata1 = pd.read_csv("dataset1/metadata_v2.tsv", header=0, sep='\t').iloc[1:]
dataset1_labels = metadata1["cell_type"]
n_clusters1 = len(np.unique(dataset1_labels))

"""Dataset 2"""
dataset2 = sc.read_10x_mtx("dataset2")
metadata2 = pd.read_csv("dataset2/metadata_v2.tsv", header=0, sep='\t').iloc[1:]
dataset2_labels = metadata2["cell_type"]
n_clusters2 = len(np.unique(dataset2_labels))


"""Dataset 3"""
dataset3 = sc.read_10x_mtx("dataset3")
metadata3 = pd.read_csv("dataset3/metadata_v2.tsv", header=0, sep='\t').iloc[1:]
dataset3_labels = metadata3["cell_type"]
n_clusters3 = len(np.unique(dataset3_labels))

"""
Datasets
A vector of tulps with as elements the dataset and the labels
"""
datasets = [(dataset1, dataset1_labels, n_clusters1),
            (dataset2, dataset2_labels, n_clusters2),
            (dataset3, dataset3_labels, n_clusters3)]
