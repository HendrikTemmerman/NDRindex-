import scanpy as sc
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

def seurat(data):
    sc.pp.normalize_total(data, target_sum=1e4)
    print("Normalized")
    return data


def log_normalization(data):
    sc.pp.normalize_total(data, target_sum=1e6)
    sc.pp.log1p(data)
    print("Normalized")
    return data


def scale(data):
    sc.pp.scale(data)
    print("Normalized")
    return data


def tmm(data):
    df = data.to_df()
    pandas2ri.activate()
    robjects.r('''
        library(edgeR)
    ''')
    edgeR = importr('edgeR')

    col_sums = df.sum(axis=0)
    df_filtered = df.loc[:, col_sums > 0]

    df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_filtered.dropna(axis=1, how='all', inplace=True)

    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        r_counts = robjects.conversion.py2rpy(df_filtered)

    dge = edgeR.DGEList(counts=r_counts)
    dge = edgeR.calcNormFactors(dge)
    norm_factors = np.array(dge.rx2('samples').rx2('norm.factors'))

    normalized_counts = df_filtered.div(norm_factors, axis=1)
    print("Normalized")

    return sc.AnnData(normalized_counts)


# TMM, Linnorm, Scale, Scarn, Seurat
normalizations = [tmm, seurat, log_normalization, scale]