import anndata
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_for_scvi(file1, file2, label_column="labels", use_basename=True,
                                  batch_key="source", n_top_genes=2000, n_latent=30):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    filters genes, prepares data for scVI, computes latent embeddings, and returns the two embedded AnnData objects.
    """

    # Load datasets
    adata1 = anndata.read_h5ad(file1)
    adata2 = anndata.read_h5ad(file2)

    # Label source
    def extract_filename(path):
        return os.path.basename(path).rsplit('.h5ad', 1)[0]

    label1 = extract_filename(file1) if use_basename else file1
    label2 = extract_filename(file2) if use_basename else file2
    adata1.obs[batch_key] = pd.Categorical([label1] * adata1.n_obs)
    adata2.obs[batch_key] = pd.Categorical([label2] * adata2.n_obs)

    # Filter new data to match label column values
    if label_column in adata1.obs and label_column in adata2.obs:
        adata2 = adata2[adata2.obs[label_column].isin(adata1.obs[label_column])].copy()

    # Concatenate for joint preprocessing
    full = anndata.concat([adata1, adata2], join="outer", label=batch_key, keys=[label1, label2])

    # Use raw counts
    full.X = full.X.astype(int)
    full.raw = full.copy()

    # Select highly variable genes
    sc.pp.highly_variable_genes(
        full,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        subset=True,
    )

    # Setup for scVI
    scvi.model.SCVI.setup_anndata(full, batch_key=batch_key)

    # Train scVI model
    model = scvi.model.SCVI(full, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    model.train()

    # Store latent embeddings
    full.obsm["X_scVI"] = model.get_latent_representation()

    #Testing the integration
    sc.pp.neighbors(full, use_rep="X_scVI")
    sc.tl.umap(full)
    sc.pl.umap(full, color=batch_key)
    sc.pl.umap(full, color=label_column) #testing via the label_column by coloring base on the label_column


    # Split back the datasets
    adata1_out = full[full.obs[batch_key] == label1].copy()
    adata2_out = full[full.obs[batch_key] == label2].copy()

    return adata1_out, adata2_out
