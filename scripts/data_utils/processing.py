import anndata
import scanpy as sc
import os
import pandas as pd  # Ensure pandas handles string assignments correctly
from sklearn import decomposition
import scipy
import numpy as np

def load_and_preprocess(file1, file2, label_column="labels", use_basename=True):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    concatenates them, filters genes, normalizes, log-transforms, standardizes,
    and computes PCA. Returns the processed AnnData object.
    """
    # Load the datasets
    adata = anndata.read_h5ad(file1)
    new = anndata.read_h5ad(file2)

    # Debug print to check file paths
    print(f"file1 path: {file1}")
    print(f"file2 path: {file2}")

    # Extract just the filename before ".h5ad"
    def extract_filename(path):
        filename = os.path.basename(path)  # Get file name
        return filename.rsplit('.h5ad', 1)[0]  # Remove the extension

    if use_basename:
        adata.obs["source"] = pd.Categorical([extract_filename(file1)] * adata.n_obs)
        new.obs["source"] = pd.Categorical([extract_filename(file2)] * new.n_obs)
    else:
        adata.obs["source"] = pd.Categorical([file1] * adata.n_obs)
        new.obs["source"] = pd.Categorical([file2] * new.n_obs)

    # Debug print to check source labels
    print("Unique source labels in adata:", adata.obs["source"].unique())
    print("Unique source labels in new:", new.obs["source"].unique())

    # Filter new data: keep only cells whose label_column values exist in adata
    if label_column in new.obs and label_column in adata.obs:
        cell_mask = new.obs[label_column].isin(adata.obs[label_column])
        new = new[cell_mask].copy()

    # Concatenate the two datasets
    full = adata.concatenate(new)

    # Filter genes with at least 1 count
    sc.pp.filter_genes(full, min_counts=1)

    # Normalize and log-transform
    adata_norm = full.copy()
    adata_norm.X = adata_norm.X.astype(float)
    sc.pp.normalize_per_cell(adata_norm, counts_per_cell_after=1_000_000)
    sc.pp.log1p(adata_norm)

    # Convert sparse matrix to dense and standardize
    adata_norm.X = adata_norm.X.toarray()
    adata_norm.X -= adata_norm.X.mean(axis=0)
    adata_norm.X /= adata_norm.X.std(axis=0)

    # Compute PCA and store in obsm
    full.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(adata_norm.X)

    return full

def load_and_preprocess_re(file1, file2, label_column="labels", use_basename=True):
    """
    Loads two AnnData files, assigns source labels, filters matching cells,
    preprocesses both datasets, and returns them separately.
    """
    # Load the datasets
    adata1 = anndata.read_h5ad(file1)
    adata2 = anndata.read_h5ad(file2)

    # Debug print to check file paths
    print(f"file1 path: {file1}")
    print(f"file2 path: {file2}")

    # Extract just the filename before ".h5ad"
    def extract_filename(path):
        filename = os.path.basename(path)  # Get file name
        return filename.rsplit('.h5ad', 1)[0]  # Remove the extension

    if use_basename:
        adata1.obs["source"] = pd.Categorical([extract_filename(file1)] * adata1.n_obs)
        adata2.obs["source"] = pd.Categorical([extract_filename(file2)] * adata2.n_obs)
    else:
        adata1.obs["source"] = pd.Categorical([file1] * adata1.n_obs)
        adata2.obs["source"] = pd.Categorical([file2] * adata2.n_obs)

    # Debug print to check source labels
    print("Unique source labels in adata1:", adata1.obs["source"].unique())
    print("Unique source labels in adata2:", adata2.obs["source"].unique())

    # Filter adata2: keep only cells whose label_column values exist in adata1
    if label_column in adata2.obs and label_column in adata1.obs:
        cell_mask = adata2.obs[label_column].isin(adata1.obs[label_column])
        adata2 = adata2[cell_mask].copy()
    
    # Use anndata.concat instead of concatenate
    full = anndata.concat([adata1, adata2])
    sc.pp.filter_genes(full, min_counts=1)
    
    # Get the list of genes to keep
    genes_to_keep = full.var_names
    
    # Process each dataset separately
    for adata in [adata1, adata2]:
        # Keep only the genes that passed filtering
        adata._inplace_subset_var(genes_to_keep)
        
        # Normalize and log-transform
        adata.X = adata.X.astype(float)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1_000_000)
        sc.pp.log1p(adata)
        
        # Convert sparse matrix to dense
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        
        # Standardize safely - avoid division by zero
        means = adata.X.mean(axis=0)
        adata.X -= means
        
        # Calculate standard deviations and handle zeros
        stds = adata.X.std(axis=0)
        # Replace zeros with 1 to avoid division by zero
        stds[stds == 0] = 1.0
        adata.X /= stds
        
        # Compute PCA, skipping features with NaNs if any remain
        # First check if there are any NaNs
        if np.isnan(adata.X).any():
            print(f"Warning: {np.isnan(adata.X).sum()} NaN values in dataset after preprocessing.")
            # You could implement additional cleaning here
        
        try:
            adata.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(adata.X)
        except ValueError as e:
            print(f"PCA failed: {e}")
            # Fallback: try to remove NaNs
            from sklearn.impute import SimpleImputer
            print("Attempting to impute NaN values...")
            imputer = SimpleImputer(strategy='mean')
            adata.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(
                imputer.fit_transform(adata.X)
            )
    
    return adata1, adata2

def load_and_preprocess_re2(file1, file2, label_column="labels", use_basename=True):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    concatenates them, filters genes, normalizes, log-transforms, standardizes,
    and computes PCA. Returns two separate processed AnnData objects.
    """
    #Best result
    # Load the datasets
    adata = anndata.read_h5ad(file1)
    new = anndata.read_h5ad(file2)

    # Debug print to check file paths
    print(f"file1 path: {file1}")
    print(f"file2 path: {file2}")

    # Extract just the filename before ".h5ad"
    def extract_filename(path):
        filename = os.path.basename(path)  # Get file name
        return filename.rsplit('.h5ad', 1)[0]  # Remove the extension

    # Store original source names for later splitting
    source1_name = extract_filename(file1) if use_basename else file1
    source2_name = extract_filename(file2) if use_basename else file2

    if use_basename:
        adata.obs["source"] = pd.Categorical([source1_name] * adata.n_obs)
        new.obs["source"] = pd.Categorical([source2_name] * new.n_obs)
    else:
        adata.obs["source"] = pd.Categorical([source1_name] * adata.n_obs)
        new.obs["source"] = pd.Categorical([source2_name] * new.n_obs)

    # Debug print to check source labels
    print("Unique source labels in adata:", adata.obs["source"].unique())
    print("Unique source labels in new:", new.obs["source"].unique())

    # Filter new data: keep only cells whose label_column values exist in adata
    if label_column in new.obs and label_column in adata.obs:
        cell_mask = new.obs[label_column].isin(adata.obs[label_column])
        new = new[cell_mask].copy()

    # Concatenate the two datasets
    full = adata.concatenate(new)

    # Filter genes with at least 1 count
    sc.pp.filter_genes(full, min_counts=1)

    # Normalize and log-transform
    adata_norm = full.copy()
    adata_norm.X = adata_norm.X.astype(float)
    sc.pp.normalize_per_cell(adata_norm, counts_per_cell_after=1_000_000)
    sc.pp.log1p(adata_norm)

    # Convert sparse matrix to dense and standardize
    adata_norm.X = adata_norm.X.toarray()
    adata_norm.X -= adata_norm.X.mean(axis=0)
    adata_norm.X /= adata_norm.X.std(axis=0)

    # Compute PCA and store in obsm
    full.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(adata_norm.X)

    # Split the processed data back into two datasets based on source
    mask_source1 = full.obs["source"] == source1_name
    mask_source2 = full.obs["source"] == source2_name
    
    adata1 = full[mask_source1].copy()
    adata2 = full[mask_source2].copy()

    return adata1, adata2

def load_and_preprocess_single(file, label_column="labels", use_basename=True):
    """
    Loads one AnnData file, assigns a source label, filters genes,
    normalizes, log-transforms, standardizes, and computes PCA.
    Returns the processed AnnData object.
    """
    # Load the dataset
    adata = anndata.read_h5ad(file)

    # Debug print to check file path
    print(f"file path: {file}")

    # Extract filename before ".h5ad"
    def extract_filename(path):
        filename = os.path.basename(path)
        return filename.rsplit('.h5ad', 1)[0]

    # Assign source label
    if use_basename:
        adata.obs["source"] = pd.Categorical([extract_filename(file)] * adata.n_obs)
    else:
        adata.obs["source"] = pd.Categorical([file] * adata.n_obs)

    # Debug print to check source label
    print("Unique source label:", adata.obs["source"].unique())

    # Filter genes with at least 1 count
    sc.pp.filter_genes(adata, min_counts=1)

    # Normalize and log-transform
    adata_norm = adata.copy()
    adata_norm.X = adata_norm.X.astype(float)
    sc.pp.normalize_per_cell(adata_norm, counts_per_cell_after=1_000_000)
    sc.pp.log1p(adata_norm)

    # Convert sparse matrix to dense and standardize
    adata_norm.X = adata_norm.X.toarray()
    adata_norm.X -= adata_norm.X.mean(axis=0)
    adata_norm.X /= adata_norm.X.std(axis=0)

    # Compute PCA and store in obsm
    adata.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(adata_norm.X)

    return adata



def load_and_preprocess_separately(file1, file2, label_column="labels", use_basename=True):
    """
    Loads two AnnData files, assigns source labels, filters matching cells based on a given column,
    filters genes, normalizes, log-transforms, standardizes, and computes PCA separately.
    Returns the two processed AnnData objects.
    """
    # Load the datasets
    adata = anndata.read_h5ad(file1)
    new = anndata.read_h5ad(file2)

    # Extract just the filename before ".h5ad"
    def extract_filename(path):
        filename = os.path.basename(path)
        return filename.rsplit('.h5ad', 1)[0]

    if use_basename:
        adata.obs["source"] = pd.Categorical([extract_filename(file1)] * adata.n_obs)
        new.obs["source"] = pd.Categorical([extract_filename(file2)] * new.n_obs)
    else:
        adata.obs["source"] = pd.Categorical([file1] * adata.n_obs)
        new.obs["source"] = pd.Categorical([file2] * new.n_obs)

    # Filter `new`: only keep cells whose label_column values are present in `adata`
    if label_column in new.obs and label_column in adata.obs:
        cell_mask = new.obs[label_column].isin(adata.obs[label_column])
        new = new[cell_mask].copy()

    def preprocess(adata):
        # Filter genes with at least 1 count
        sc.pp.filter_genes(adata, min_counts=1)

        # Normalize and log-transform
        adata.X = adata.X.astype(float)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1_000_000)
        sc.pp.log1p(adata)

        # Convert sparse matrix to dense (if needed) and standardize
        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()
        adata.X -= adata.X.mean(axis=0)
        adata.X /= adata.X.std(axis=0)

        # Compute PCA
        adata.obsm["X_pca"] = decomposition.PCA(n_components=50).fit_transform(adata.X)
        return adata

    adata_processed = preprocess(adata)
    new_processed = preprocess(new)

    return adata_processed, new_processed



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

    # Split back the datasets
    adata1_out = full[full.obs[batch_key] == label1].copy()
    adata2_out = full[full.obs[batch_key] == label2].copy()

    return adata1_out, adata2_out
