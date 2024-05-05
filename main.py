import scanpy as sc
import copy as cp
from Datasets import datasets
from Normalization import normalizations
from DimensionalityReduction import dimension_reductions
import numpy as np
#from NDRindex import NDRindex
from newNDRindex import NDRindex
from ARI import calculated_ARI

counter = 1

for dataset, true_labels in datasets:
    for normalize in normalizations:
        for reduce_dimension in dimension_reductions:
            print("------------------", str(normalize.__name__).capitalize(), "---------------", str(reduce_dimension.__name__).capitalize(), "---------------")
            adata = cp.deepcopy(dataset)
            normalize(adata)
            sc.pp.neighbors(adata, use_rep='X')
            reduce_dimension(adata)

            # If you just want the raw data matrix (genes x cells)
            data_matrix = adata.X

            # If the data matrix is sparse, convert it to a dense format
            if isinstance(data_matrix, np.ndarray):
                dense_matrix = data_matrix
            else:
                dense_matrix = data_matrix.toarray()

            # Now, if you have already performed some dimensionality reduction and it is stored in .obsm
            # For example, if you have PCA results in .obsm['X_pca']
            dimension_reduction_method = reduce_dimension.__name__
            for col in adata.obsm.keys():
                if dimension_reduction_method in col:
                    reduced_data_matrix = adata.obsm[col]
                    ndr_input = reduced_data_matrix
                else:
                    ndr_input = dense_matrix


            # Now you can pass ndr_input to the NDRindex function
            print("NDRindex for dataset", counter, ": ",  normalize.__name__, " and ", dimension_reduction_method, ": ", NDRindex(ndr_input, true_labels), " and the ARI is equal to ", calculated_ARI(ndr_input, true_labels))
    counter += 1
