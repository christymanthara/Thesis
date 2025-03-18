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
    adata = anndata.read_h5ad(path.join("..", "datasets", file1))
    new = anndata.read_h5ad(path.join("..", "datasets", file2))
    
    adata.obs["source"] = "Dataset 1"
    new.obs["source"] = "Dataset 2"
    
    cell_mask = new.obs["labels"].isin(adata.obs["labels"])
    new = new[cell_mask].copy()
    
    full = adata.concatenate(new)
    sc.pp.filter_genes(full, min_counts=1)
    
    adata_norm = full.copy()
    sc.pp.normalize_per_cell(adata_norm, counts_per_cell_after=1_000_000)
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
    
    plt.figure(figsize=(8, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=full.obs["source"].astype("category").cat.codes, cmap="viridis", alpha=0.7)
    plt.colorbar(label="Dataset Source")
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(output_pdf)
    plt.close()
