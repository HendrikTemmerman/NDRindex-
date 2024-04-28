
import pandas as pd
import numpy as np
from rpy2 import robjects
from rpy2.robjects import pandas2ri
def total_count_normalization(data):
    # Convert Anndata object to DataFrame
    adata_df = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)

    # Pass the DataFrame to R
    robjects.globalenv['adata_df'] = pandas2ri.py2rpy(adata_df)

    # Your R code
    robjects.r('''
        library(scanpy)

        # Convert the DataFrame back to an Anndata object
        adata <- as.list(adata_df)
        adata <- AnnData(X = adata$X, obs = adata$obs, var = adata$var)

        # Perform total count normalization
        sc.pp.normalize_total(adata, target_sum=1e6)

        # Extract the normalized data
        normalized_data <- adata$X

        # Return the normalized data
        result <- normalized_data
    ''')

    # Retrieve results back to Python
    normalized_data = robjects.globalenv['result']

    # Update the data with the normalized values
    data.X = np.array(normalized_data)

    return data



def log_normalization(data):
    # Convert Anndata object to DataFrame
    adata_df = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)

    # Pass the DataFrame to R
    robjects.globalenv['adata_df'] = pandas2ri.py2rpy(adata_df)

    # Your R code
    robjects.r('''
        library(scanpy)

        # Convert the DataFrame back to an Anndata object
        adata <- as.list(adata_df)
        adata <- AnnData(X = adata$X, obs = adata$obs, var = adata$var)

        # Perform total count normalization
        sc.pp.normalize_total(adata, target_sum=1e6)

        # Perform logarithmic transformation
        sc.pp.log1p(adata)

        # Extract the normalized and log-transformed data
        normalized_log_data <- adata$X

        # Return the normalized and log-transformed data
        result <- normalized_log_data
    ''')

    # Retrieve results back to Python
    normalized_log_data = robjects.globalenv['result']

    # Update the data with the normalized and log-transformed values
    data.X = np.array(normalized_log_data)

    return data


def scale_normalization(data):
    # Convert Anndata object to DataFrame
    adata_df = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)

    # Pass the DataFrame to R
    robjects.globalenv['adata_df'] = pandas2ri.py2rpy(adata_df)

    # Your R code
    robjects.r('''
        library(scanpy)

        # Convert the DataFrame back to an Anndata object
        adata <- as.list(adata_df)
        adata <- AnnData(X = adata$X, obs = adata$obs, var = adata$var)

        # Perform scaling normalization
        sc.pp.scale(adata)

        # Extract the scaled data
        scaled_data <- adata$X

        # Return the scaled data
        result <- scaled_data
    ''')

    # Retrieve results back to Python
    scaled_data = robjects.globalenv['result']

    # Update the data with the scaled values
    data.X = np.array(scaled_data)

    return data

normalizations =[total_count_normalization]


"""


import scanpy as sc


def total_count_normalization(data):
    sc.pp.normalize_total(data, target_sum=1e6)


def log_normalization(data):
    sc.pp.normalize_total(data, target_sum=1e6)
    sc.pp.log1p(data)


def scale_normalization(data):
    sc.pp.scale(data)


#normalizations = [total_count_normalization, log_normalization, scale_normalization]
normalizations = [total_count_normalization]
"""