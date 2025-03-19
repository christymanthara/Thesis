from os import path
import sys
sys.path.append(path.join("..", "notebooks"))
sys.path.append(path.join("utils.py"))
import utils

import openTSNE
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import string
import gzip
import pickle
from sklearn import decomposition
import matplotlib.pyplot as plt

def tsne_pavlin(file1, file2, output_pdf="tsne_plot.pdf"):
    # adata = anndata.read_h5ad(path.join("..", "datasets", file1))
    # new = anndata.read_h5ad(path.join("..", "datasets", file2))
    
    """
    Loads two AnnData files, merges them, computes tsne, and saves the plot as a PDF.
    
    Parameters:
    - file1 (str): Path to the first .h5ad file.
    - file2 (str): Path to the second .h5ad file.
    - output_file (str): Path where the tsne plot should be saved.
    """

    # Load the datasets
    adata = anndata.read_h5ad(file1)
    new = anndata.read_h5ad(file2)

    adata.obs["source"] = file1
    new.obs["source"] = file2
    
    cell_mask = new.obs["labels"].isin(adata.obs["labels"]) #to be added to the arguments
    new = new[cell_mask].copy()
    
    full = adata.concatenate(new)
    sc.pp.filter_genes(full, min_counts=1)
    
    adata_norm = full.copy()
    sc.pp.normalize_per_cell(adata_norm, counts_per_cell_after=1_000_000) #add to argument
    sc.pp.log1p(adata_norm)
    
    adata_norm.X = adata_norm.X.toarray()
    adata_norm.X -= adata_norm.X.mean(axis=0)
    adata_norm.X /= adata_norm.X.std(axis=0)
    
    full.obsm["pca"] = decomposition.PCA(n_components=50).fit_transform(adata_norm.X)
    
    affinities = openTSNE.affinity.Multiscale(
        full.obsm["pca"],
        perplexities=[50, 500],
        metric="cosine",
        n_jobs=8,
        random_state=0,
    )
    init = openTSNE.initialization.pca(full.obsm["pca"], random_state=0)
    embedding = openTSNE.TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=8,
    )
    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
    
    # utils.plot(embedding, full.obs["source"])
    # plt.savefig("output_plot.pdf", format="pdf", bbox_inches="tight")
    # plt.close()
    utils.plot(embedding, full.obs["source"],save_path="tsne_plot.pdf")


if __name__ == "__main__":
    tsne_pavlin("../datasets/baron_2016h.h5ad", "../datasets/xin_2016.h5ad")