import os
import numpy as np
import pickle
import hashlib
from pathlib import Path

def get_dataset_fingerprint(adata, fingerprint_column='source'):
    """
    Generate a unique fingerprint for the dataset based on unique values in a column.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object
    fingerprint_column : str, default 'source'
        Column name to use for fingerprinting (e.g., 'source', 'batch', etc.)
    
    Returns:
    --------
    str : A hash string representing the dataset fingerprint
    """
    if fingerprint_column not in adata.obs.columns:
        print(f"Warning: {fingerprint_column} column not found. Using dataset shape as fingerprint.")
        # Fallback to using dataset dimensions and a sample of gene names
        fingerprint_data = f"{adata.n_obs}_{adata.n_vars}_{hash(tuple(adata.var_names[:100].tolist()))}"
    else:
        # Get unique values and sort them for consistency
        unique_values = sorted(adata.obs[fingerprint_column].unique().tolist())
        # Also include the count of each unique value for more specificity
        value_counts = adata.obs[fingerprint_column].value_counts().sort_index()
        fingerprint_data = f"{unique_values}_{value_counts.tolist()}_{adata.n_obs}_{adata.n_vars}"
    
    # Create a hash of the fingerprint data
    fingerprint_hash = hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]  # Use first 12 chars
    return fingerprint_hash

def get_embedding_filename(dataset_name, embedding_type, dataset_fingerprint=None, **params):
    """Generate standardized embedding filename with optional parameters and dataset fingerprint"""
    param_str = "_".join([f"{k}{v}" for k, v in params.items()])
    filename = f"{dataset_name}_{embedding_type}"
    if param_str:
        filename += f"_{param_str}"
    if dataset_fingerprint:
        filename += f"_fp{dataset_fingerprint}"
    return filename + ".pkl"

def save_embedding(embedding, dataset_name, embedding_type, embedding_dir="embeddings", 
                  dataset_fingerprint=None, **params):
    """Save embedding to file with dataset fingerprint"""
    Path(embedding_dir).mkdir(exist_ok=True)
    filename = get_embedding_filename(dataset_name, embedding_type, dataset_fingerprint, **params)
    filepath = Path(embedding_dir) / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)
    print(f"Saved embedding: {filepath}")

def load_embedding(dataset_name, embedding_type, embedding_dir="embeddings", 
                  dataset_fingerprint=None, **params):
    """Load embedding if exists, return None otherwise"""
    filename = get_embedding_filename(dataset_name, embedding_type, dataset_fingerprint, **params)
    filepath = Path(embedding_dir) / filename
    
    if filepath.exists():
        with open(filepath, 'rb') as f:
            embedding = pickle.load(f)
        print(f"Loaded existing embedding: {filepath}")
        return embedding
    else:
        print(f"No cached embedding found for: {filename}")
    return None

def cleanup_old_embeddings(dataset_name, embedding_type, embedding_dir="embeddings", 
                          current_fingerprint=None, **params):
    """
    Remove old embedding files for the same dataset/parameters but different fingerprints.
    This helps keep the cache directory clean.
    """
    embedding_dir = Path(embedding_dir)
    if not embedding_dir.exists():
        return
    
    # Create pattern to match old embeddings
    param_str = "_".join([f"{k}{v}" for k, v in params.items()])
    base_pattern = f"{dataset_name}_{embedding_type}"
    if param_str:
        base_pattern += f"_{param_str}"
    
    # Find all files that match the base pattern
    pattern = f"{base_pattern}_fp*.pkl"
    matching_files = list(embedding_dir.glob(pattern))
    
    # Remove files that don't match the current fingerprint
    current_filename = get_embedding_filename(dataset_name, embedding_type, current_fingerprint, **params)
    
    for file_path in matching_files:
        if file_path.name != current_filename:
            print(f"Removing old embedding cache: {file_path}")
            file_path.unlink()

def compute_or_load_embedding(adata, dataset_name, embedding_type, embedding_function, 
                             embedding_dir="embeddings", force_recompute=False, 
                             fingerprint_column='source', cleanup_old=True, **embedding_params):
    """
    Generalized function to compute or load embeddings with dataset fingerprinting.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object
    dataset_name : str
        Name of the dataset
    embedding_type : str
        String identifier for embedding type, e.g., 'scVI', 'scANVI'
    embedding_function : callable
        Function that computes embedding
    embedding_dir : str, default "embeddings"
        Directory to save/load embeddings
    force_recompute : bool, default False
        If True, always recompute embedding
    fingerprint_column : str, default 'source'
        Column name to use for dataset fingerprinting
    cleanup_old : bool, default True
        Whether to remove old embedding files with different fingerprints
    **embedding_params : dict
        Parameters passed both for filename and to embedding_function
    
    Returns:
    --------
    Computed or loaded embedding
    """
    # Generate dataset fingerprint
    dataset_fingerprint = get_dataset_fingerprint(adata, fingerprint_column)
    print(f"Dataset fingerprint ({fingerprint_column}): {dataset_fingerprint}")
    
    if not force_recompute:
        existing_embedding = load_embedding(
            dataset_name, embedding_type, embedding_dir, 
            dataset_fingerprint, **embedding_params
        )
        if existing_embedding is not None:
            return existing_embedding
    
    # Clean up old embeddings if requested
    if cleanup_old:
        cleanup_old_embeddings(
            dataset_name, embedding_type, embedding_dir, 
            dataset_fingerprint, **embedding_params
        )
    
    # Compute new embedding
    print(f"Computing {embedding_type} for {dataset_name}...")
    embedding = embedding_function(adata, **embedding_params)
    
    save_embedding(
        embedding, dataset_name, embedding_type, embedding_dir, 
        dataset_fingerprint, **embedding_params
    )
    return embedding

# Additional utility function to check cache status
def check_embedding_cache_status(adata, dataset_name, embedding_type, embedding_dir="embeddings", 
                                fingerprint_column='source', **embedding_params):
    """
    Check if a valid cached embedding exists for the current dataset.
    
    Returns:
    --------
    dict : Status information including whether cache exists and fingerprint info
    """
    dataset_fingerprint = get_dataset_fingerprint(adata, fingerprint_column)
    filename = get_embedding_filename(dataset_name, embedding_type, dataset_fingerprint, **embedding_params)
    filepath = Path(embedding_dir) / filename
    
    status = {
        'cache_exists': filepath.exists(),
        'cache_path': str(filepath),
        'dataset_fingerprint': dataset_fingerprint,
        'fingerprint_column': fingerprint_column
    }
    
    if fingerprint_column in adata.obs.columns:
        status['unique_values'] = sorted(adata.obs[fingerprint_column].unique().tolist())
        status['value_counts'] = adata.obs[fingerprint_column].value_counts().to_dict()
    
    return status