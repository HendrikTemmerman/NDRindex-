import scanpy as sc

"""
Dimension reduction methode: PCA
The PCA method will reduce the gene expression matrix to its principal components. 
The principal components are orthogonal to each other and capture the essence of the variation in the gene expressions.
"""
def pca(data):
    sc.pp.pca(data, svd_solver='arpack')


"""
Dimension reduction methode: t-SNE
The t-<sne method will transform similarities between data points to joint a probabilities. 
Then, t-SNE will minimize the divergence between the probabilities of the low-dimensional points and the high-dimensional data.
"""
def tsne(data):
    sc.tl.tsne(data, n_pcs=50)

"""
Dimension reduction methode: UMAP
The method UMAP will construct a graph using the high-dimensional data and optimize a low-dimensional graph to best match the topology
"""
def umap(data):
    sc.pp.neighbors(data, n_pcs=50, use_rep='X')
    sc.tl.umap(data)

"""
Dimension reduction methode: diffmap
The diffmap method will use diffusion maps to map the high-dimensional data to a low-dimension space.
"""
def diffmap(data):
    sc.tl.diffmap(data)

"""The list of all dimension reductions we use in the experiment"""
dimension_reductions = [pca, tsne, umap, diffmap]
