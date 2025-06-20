import os
import numpy as np
import pickle
from pathlib import Path

def get_embedding_filename(dataset_name, embedding_type, **params):
    """Generate standardized embedding filename with optional parameters"""
    param_str = "_".join([f"{k}{v}" for k, v in params.items()])
    filename = f"{dataset_name}_{embedding_type}"
    if param_str:
        filename += f"_{param_str}"
    return filename + ".pkl"

def save_embedding(embedding, dataset_name, embedding_type, embedding_dir="embeddings", **params):
    """Save embedding to file"""
    Path(embedding_dir).mkdir(exist_ok=True)
    filename = get_embedding_filename(dataset_name, embedding_type, **params)
    filepath = Path(embedding_dir) / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)
    print(f"Saved embedding: {filepath}")

def load_embedding(dataset_name, embedding_type, embedding_dir="embeddings", **params):
    """Load embedding if exists, return None otherwise"""
    filename = get_embedding_filename(dataset_name, embedding_type, **params)
    filepath = Path(embedding_dir) / filename
    
    if filepath.exists():
        with open(filepath, 'rb') as f:
            embedding = pickle.load(f)
        print(f"Loaded existing embedding: {filepath}")
        return embedding
    return None

def compute_or_load_embedding(adata, dataset_name, embedding_type, embedding_function, embedding_dir="embeddings", force_recompute=False, **embedding_params):
    """
    Generalized function to compute or load embeddings.
    
    Parameters:
        adata: AnnData object
        dataset_name: name of the dataset
        embedding_type: string, e.g., 'X_pca', 'X_tsne'
        embedding_function: callable, function that computes embedding
        embedding_dir: directory to save/load embeddings
        force_recompute: if True, always recompute embedding
        **embedding_params: parameters passed both for filename and to embedding_function
    """
    if not force_recompute:
        existing_embedding = load_embedding(dataset_name, embedding_type, embedding_dir, **embedding_params)
        if existing_embedding is not None:
            return existing_embedding

    # Compute new embedding
    print(f"Computing {embedding_type} for {dataset_name}...")
    embedding = embedding_function(adata, **embedding_params)
    
    save_embedding(embedding, dataset_name, embedding_type, embedding_dir, **embedding_params)
    return embedding



