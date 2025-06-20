import os
import numpy as np
import pickle
from pathlib import Path

def get_embedding_filename(dataset_name, embedding_type, n_components=None):
    """Generate standardized embedding filename"""
    if n_components:
        return f"{dataset_name}_{embedding_type}_{n_components}components.pkl"
    else:
        return f"{dataset_name}_{embedding_type}.pkl"

def save_embedding(embedding, dataset_name, embedding_type, embedding_dir="embeddings", **kwargs):
    """Save embedding to file"""
    Path(embedding_dir).mkdir(exist_ok=True)
    filename = get_embedding_filename(dataset_name, embedding_type, **kwargs)
    filepath = Path(embedding_dir) / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)
    print(f"Saved embedding: {filepath}")

def load_embedding(dataset_name, embedding_type, embedding_dir="embeddings", **kwargs):
    """Load embedding if exists, return None otherwise"""
    filename = get_embedding_filename(dataset_name, embedding_type, **kwargs)
    filepath = Path(embedding_dir) / filename
    
    if filepath.exists():
        with open(filepath, 'rb') as f:
            embedding = pickle.load(f)
        print(f"Loaded existing embedding: {filepath}")
        return embedding
    return None

# Usage example:
def compute_or_load_pca(adata, dataset_name, n_components=50):
    """Compute PCA or load if already exists"""
    
    # Try to load existing embedding
    pca_embedding = load_embedding(dataset_name, "X_pca", n_components=n_components)
    
    if pca_embedding is not None:
        return pca_embedding
    
    # Compute new embedding
    print(f"Computing PCA for {dataset_name}...")
    from sklearn.decomposition import PCA
    pca_embedding = PCA(n_components=n_components).fit_transform(adata.X)
    
    # Save for future use
    save_embedding(pca_embedding, dataset_name, "X_pca", n_components=n_components)
    
    return pca_embedding

# Use it:
baron_pca = compute_or_load_pca(baron_adata, "baron_2016", n_components=50)
baron_adata.obsm["X_pca"] = baron_pca