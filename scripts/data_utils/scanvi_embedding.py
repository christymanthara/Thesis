import anndata
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
num_workers = multiprocessing.cpu_count()

def load_and_preprocess_for_scanvi(file1, file2, label_column="labels", use_basename=True,
                                   batch_key="source", n_top_genes=2000, n_latent=30):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    filters genes, prepares data for scANVI, computes latent embeddings, and returns the two embedded AnnData objects.
    
    This version uses scANVI for semi-supervised integration rather than scVI.
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

    # Make sure the label column is categorical
    if label_column in full.obs:
        full.obs[label_column] = full.obs[label_column].astype('category')
    
    # IMPORTANT: Create a fresh copy of the AnnData object before setup
    # This avoids the "setup with a different model" error
    full_copy = full.copy()
    
    # Setup for scANVI directly - don't reuse AnnData that was set up with a different model
    scvi.model.SCANVI.setup_anndata(
        full_copy, 
        batch_key=batch_key,
        labels_key=label_column,
        unlabeled_category="Unknown"  # Category for unlabeled cells if any
    )

    # First train a scVI model as initialization for scANVI

    scvi.model.SCVI.setup_anndata(full_copy, batch_key=batch_key)
    vae = scvi.model.SCVI(full_copy, n_layers=2, n_latent=n_latent, gene_likelihood="nb")
    vae.train()
    
    # Initialize and train the scANVI model
    model = scvi.model.SCANVI.from_scvi_model(vae,adata =full_copy,labels_key=label_column, unlabeled_category="Unknown")
    model.train(max_epochs=20)  # Typically needs fewer epochs when initialized from scVI

    # Store latent embeddings
    full_copy.obsm["X_scANVI"] = model.get_latent_representation()

    # Testing the integration
    sc.pp.neighbors(full_copy, use_rep="X_scANVI")
    sc.tl.umap(full_copy)
    sc.pl.umap(full_copy, color=batch_key)
    sc.pl.umap(full_copy, color=label_column)  # Testing via the label_column

    # Split back the datasets
    adata1_out = full_copy[full_copy.obs[batch_key] == label1].copy()
    adata2_out = full_copy[full_copy.obs[batch_key] == label2].copy()

    return adata1_out, adata2_out