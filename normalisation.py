import scanpy as sc
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

"""
Normalisation methode: total
The methode total will scale the gene expression counts in each cell to ensure that the total counts per cell are equalized. 
This results in gene expression values that can be compared across the samples
"""


def total(data):
    sc.pp.normalize_total(data, target_sum=1e4)
    return data


"""
Normalisation methode: log_normalization
The log_normalization will perform total count normalization and subsequently logarithmise the values
"""


def log_normalization(data):
    sc.pp.normalize_total(data, target_sum=1e6)
    sc.pp.log1p(data)
    return data


"""
Normalisation methode: scale
The scale method will address noise by scaling the gene expression levels to unit variance and a mean of zero.
"""


def scale(data):
    sc.pp.scale(data)
    return data


"""
Normalisation methode: tmm
The methode tmm will address the differences in gene expression levels by computing the M-value or log-ratio for
each pair of samples. These values are then trimmed to reduce the impact of outliers. 
Finally, a scaling factor is computed using the M-values, to normalize the samples.
For using the methode tmm we used rpy2 to access R code in python.
"""


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
    dge = edgeR.calcNormFactors(dge, method="TMM")
    norm_factors = np.array(dge.rx2('samples').rx2('norm.factors'))

    normalized_counts = df_filtered.div(norm_factors, axis=1)
    return sc.AnnData(normalized_counts)


"""The list of all normalisations we use in the experiment"""
normalizations = [tmm, total, log_normalization, scale]
