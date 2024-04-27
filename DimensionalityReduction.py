import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import pandas as pd



def pca(data):

    # Assuming 'adata' is your Anndata object
    adata_df = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)
    # Pass the DataFrame to R
    robjects.globalenv['adata_df'] = pandas2ri.py2rpy(adata_df)

    # Your R code
    robjects.r('''
        library(irlba)

        # Perform PCA using irlba
        pca_result <- irlba(t(adata_df), nv = min(nrow(adata_df), ncol(adata_df)), center = TRUE, scale = TRUE)

        # Extract PCA results
        pca_scores <- pca_result$u
        pca_loadings <- pca_result$v

        # Return PCA scores and loadings
        pca_results <- list(scores = pca_scores, loadings = pca_loadings)
    ''')

    # Retrieve PCA results back to Python
    pca_scores = robjects.globalenv['pca_results'].rx2('scores')
    pca_loadings = robjects.globalenv['pca_results'].rx2('loadings')
    return pca_scores, pca_loadings


def tsne(data):
    # Convert Anndata object to DataFrame
    adata_df = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)

    # Pass the DataFrame to R
    robjects.globalenv['adata_df'] = pandas2ri.py2rpy(adata_df)

    # Your R code
    robjects.r('''
        library(scanpy)

        # Perform t-SNE using Scanpy
        sc.tl.tsne(data, n_pcs = 50)

        # Extract t-SNE results
        tsne_coords <- data$obsm[['X_tsne']]

        # Return t-SNE coordinates
        result <- tsne_coords
    ''')

    # Retrieve results back to Python
    tsne_coords = robjects.globalenv['result']

    return tsne_coords





def umap(data):
    # Convert Anndata object to DataFrame
    adata_df = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)

    # Pass the DataFrame to R
    robjects.globalenv['adata_df'] = pandas2ri.py2rpy(adata_df)

    # Your R code
    robjects.r('''
        library(umap)
        library(scanpy)

        # Compute neighbors using Scanpy
        sc.pp.neighbors(data, n_pcs = 50)

        # Perform UMAP using Scanpy
        sc.tl.umap(data)

        # Extract UMAP coordinates
        umap_coords <- data$obsm[['X_umap']]

        # Return UMAP coordinates
        result <- umap_coords
    ''')

    # Retrieve results back to Python
    umap_coords = robjects.globalenv['result']

    return umap_coords





def diffmap(data):
    # Convert Anndata object to DataFrame
    adata_df = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)

    # Pass the DataFrame to R
    robjects.globalenv['adata_df'] = pandas2ri.py2rpy(adata_df)

    # Your R code
    robjects.r('''
        library(scanpy)

        # Perform Diffmap using Scanpy
        sc.tl.diffmap(data)

        # Extract Diffmap coordinates
        diffmap_coords <- data$obsm[['X_diffmap']]

        # Return Diffmap coordinates
        result <- diffmap_coords
    ''')

    # Retrieve results back to Python
    diffmap_coords = robjects.globalenv['result']

    return diffmap_coords




dimension_reductions = [pca]





































"""
import scanpy as sc



def pca(data):
    sc.pp.pca(data, svd_solver='arpack')


def tsne(data):
    sc.tl.tsne(data, n_pcs=50)


def umap(data):
    sc.pp.neighbors(data, n_pcs=50)
    sc.tl.umap(data,)


def diffmap(data):
    sc.tl.diffmap(data)




dimension_reductions = [umap, diffmap, pca, tsne]
dimension_reductions = [pca]"""