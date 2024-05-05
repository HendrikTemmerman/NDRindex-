from sammon import sammon
import scanpy as sc
from sklearn.manifold import TSNE
from scipy.sparse import issparse


def pca(data):
    print("Before reduction:", data.shape)
    sc.pp.pca(data, svd_solver='arpack')
    print("Reduced by pca:", data.obsm['X_pca'].shape)


def tsne(data):
    print("Before reduction:", data.shape)
    sc.tl.tsne(data, n_pcs=50)
    print("Reduced by tsne:", data.obsm["X_tsne"].shape)


def tsne2(data):
    x = data.X
    print("Before reduction:", x.shape)
    reduced_x = TSNE(n_components=2, perplexity=20, init="random").fit_transform(x)
    data.obsm["X_tsne2"] = reduced_x
    print("Reduced by tsne2", reduced_x.shape)


def umap(data):
    print("Before reduction:", data.shape)
    sc.pp.neighbors(data, n_pcs=50, use_rep='X')
    sc.tl.umap(data)
    print("Reduced by umap:", data.obsm["X_umap"].shape)


def diffmap(data):
    print("Before reduction:", data.shape)
    sc.tl.diffmap(data)
    print("Reduced by diffmap:", data.obsm["X_diffmap"].shape)


def sammon_reduction(data):
    print("Before reduction:", data.shape)
    if issparse(data.X):
        x = data.X.toarray()
    else:
        x = data.X
    reduced_x, stress = sammon.sammon(x)
    data.obsm["X_sammon_reduction"] = reduced_x
    print("Reduced by sammon:", data.obsm["X_sammon_reduction"].shape)


dimension_reductions = [pca, sammon_reduction, tsne, tsne2, umap, diffmap]
