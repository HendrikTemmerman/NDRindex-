# Dataset 1: MD704
# Dataset 2: MD710
# Dataset 3: MD711


import scanpy as sc
import pandas as pd


dataset1 = sc.read_10x_mtx("dataset1")
metadata1 = pd.read_csv("dataset1/metadata_v2.tsv", header=0, sep='\t').iloc[1:]
metadata1.reset_index(drop=True, inplace=True)

dataset2 = sc.read_10x_mtx("dataset2")
metadata2 = pd.read_csv("dataset2/metadata_v2.tsv", header=0, sep='\t').iloc[1:]
metadata2.reset_index(drop=True, inplace=True)

dataset3 = sc.read_10x_mtx("dataset3")
metadata3 = pd.read_csv("dataset3/metadata_v2.tsv", header=0, sep='\t').iloc[1:]
metadata3.reset_index(drop=True, inplace=True)


datasets = [dataset1, dataset2, dataset3]
