# src/globals.py

"""
In this script, we will calculate the following global variables:

1) Sample(dataset citation) UMAP embeddings (dim 1 and dim 2). The resulting dataset will have the shape: 487 (n_samples) x 2 (dim 1 and dim 2) if working on the training set. 

2) Sample(dataset citation) PCA loadings (PC1 and PC2). The resulting dataset will have the shape: 487 (n_samples) x 2 (PC1 and PC2) if working on the training set. 

3) Feature UMAP embeddings (dim 1 and dim 2). The resulting dataset will have the shape: 384 (n_features) x 2 (dim 1 and dim 2). Can ONLY be run on training set. However, output be used to retrieve relevant context from testing set. Run once on train set and save output to file. 

4) Feature PCA loadings (PC1 and PC2). The resulting dataset will have the shape: 384 (n_features) x 2 (PC1 and PC2). Can ONLY be run on training set. However, output can be used to retrieve relevant context from testing set. Run once on train set and save output to file. 

"""