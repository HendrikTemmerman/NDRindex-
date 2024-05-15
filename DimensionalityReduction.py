from sammon import sammon
import scanpy as sc
from scipy.sparse import issparse


def pca(data):
    print("Starting dimensionality reduction...")
    print("Before reduction:", data.shape)
    sc.pp.pca(data, svd_solver='arpack')
    print("Reduced by pca:", data.obsm['X_pca'].shape)


def tsne(data):
    print("Starting dimensionality reduction...")
    print("Before reduction:", data.shape)
    sc.tl.tsne(data, n_pcs=50)
    print("Reduced by tsne:", data.obsm["X_tsne"].shape)


def umap(data):
    print("Starting dimensionality reduction...")
    print("Before reduction:", data.shape)
    sc.pp.neighbors(data, n_pcs=50, use_rep='X')
    sc.tl.umap(data)
    print("Reduced by umap:", data.obsm["X_umap"].shape)


def diffmap(data):
    print("Starting dimensionality reduction...")
    print("Before reduction:", data.shape)
    sc.tl.diffmap(data)
    print("Reduced by diffmap:", data.obsm["X_diffmap"].shape)

dimension_reductions = [pca, tsne, umap, diffmap]

