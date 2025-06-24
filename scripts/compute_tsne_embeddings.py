from openTSNE import TSNEEmbedding, affinity, initialization
import openTSNE
import numpy as np

# this is pavlins method with mutltiaffinities and PCA initialization
def compute_tsne_embedding_pavlin(
    adata, 
    embedding_key="pca", 
    output_key="tsne_pavlin",
    n_jobs=8
):
    """
    Computes t-SNE embeddings from a specified embedding in an AnnData object and stores the result in `obsm`.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the input embeddings in `obsm[embedding_key]`.
    embedding_key : str, optional
        The key in `adata.obsm` where the input embeddings are stored. Default is "pca".
    output_key : str, optional
        The key under which to store the computed t-SNE embeddings in `adata.obsm`. Default is "tsne_pavlin".

    Returns
    -------
    AnnData
        The AnnData object with the t-SNE embeddings stored in `obsm[output_key]`.
    """
    print(f"Computing t-SNE embeddings with pavlins method from {embedding_key} and storing in {output_key}...")
    print(f"Computing t-SNE embeddings from {embedding_key}...")
    print(f"Computing t-SNE for {adata.shape[1]} genes")
    affinities = affinity.Multiscale(
        adata.obsm[embedding_key], 
        perplexities=[50, 500], 
        metric="cosine", 
        n_jobs=n_jobs, 
        random_state=3
    )

    init = initialization.pca(adata.obsm[embedding_key], random_state=42)

    embedding = TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
    )

    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

    # output_key = f"{output_key}_tsne"

    adata.obsm[output_key] = embedding
    return adata

def compute_tsne_transfer_combined_pavlin(
    adata_combined, 
    reference_source,        # Name of reference source in adata.obs['source']
    target_source,           # Name of target source in adata.obs['source'] 
    embedding_key="X_pca",   # Input embedding key
    output_key="tsne_transfer",
    n_jobs=8,
    k=10
):
    """
    Computes t-SNE embedding transfer when reference and target data are in the same AnnData object.
    
    Parameters
    ----------
    adata_combined : AnnData
        Combined dataset with both reference and target data
    reference_source : str
        Value in adata.obs['source'] that identifies the reference dataset
    target_source : str  
        Value in adata.obs['source'] that identifies the target dataset
    embedding_key : str
        Key in obsm containing input embeddings (e.g., 'X_pca')
    output_key : str
        Key to store the transferred embeddings
    n_jobs : int
        Number of parallel jobs
    k : int
        Number of nearest neighbors for partial embedding
        
    Returns
    -------
    AnnData
        The combined dataset with transferred embeddings in obsm[output_key]
    """
    
    print(f"Computing embedding transfer from {reference_source} to {target_source}")
    
    # Step 1: Split the combined data
    reference_mask = adata_combined.obs['source'] == reference_source
    target_mask = adata_combined.obs['source'] == target_source
    
    print(f"Reference cells: {reference_mask.sum()}")
    print(f"Target cells: {target_mask.sum()}")
    
    # Extract reference and target data
    adata_ref = adata_combined[reference_mask].copy()
    adata_target = adata_combined[target_mask].copy()
    
    # Step 2: Compute t-SNE embedding on reference data ONLY
    print("Computing reference t-SNE embedding...")
    affinities_ref = affinity.Multiscale(
        adata_ref.obsm[embedding_key], 
        perplexities=[50, 500], 
        metric="cosine", 
        n_jobs=n_jobs, 
        random_state=3
    )
    
    init_ref = initialization.pca(adata_ref.obsm[embedding_key], random_state=42)
    
    embedding_ref = TSNEEmbedding(
        init_ref,
        affinities_ref,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
    )
    
    # Optimize reference embedding
    embedding_ref.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding_ref.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)
    
    # Step 3: Transfer target data into reference embedding space
    print("Transferring target data into reference embedding space...")
    
    # Create affinities for the reference data (for transfer)
    transfer_affinities = affinity.PerplexityBasedNN(
        adata_ref.obsm[embedding_key],
        perplexity=30,
        metric="cosine", 
        n_jobs=n_jobs,
        random_state=3
    )
    
    # Create embedding object with reference embedding
    transfer_embedding = TSNEEmbedding(
        np.array(embedding_ref),  # Use reference embedding as template
        transfer_affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
    )
    
    # Project target data into reference space
    target_embedding = transfer_embedding.prepare_partial(
        adata_target.obsm[embedding_key], 
        k=k
    )
    
    # Optimize target embedding in reference space
    target_embedding.optimize(250, learning_rate=0.1, momentum=0.8, inplace=True)
    
    # Step 4: Combine embeddings back into original order
    print("Combining embeddings...")
    
    # Initialize output array
    combined_embedding = np.zeros((adata_combined.n_obs, 2))
    
    # Fill in reference embeddings
    combined_embedding[reference_mask] = np.array(embedding_ref)
    
    # Fill in target embeddings  
    combined_embedding[target_mask] = np.array(target_embedding)
    
    # Store in combined dataset
    adata_combined.obsm[output_key] = combined_embedding
    
    print(f"✅ Transfer complete! Embeddings stored in obsm['{output_key}']")
    
    return adata_combined


def compute_tsne_standard_combined(
    adata_combined,
    embedding_key="X_pca", 
    output_key="tsne_standard",
    n_jobs=8
):
    """
    Computes standard t-SNE embedding on combined dataset (no transfer).
    This treats all data as one dataset and computes embedding jointly.
    """
    print(f"Computing standard t-SNE embedding on combined dataset...")
    print(f"Total cells: {adata_combined.n_obs}")
    
    affinities = affinity.Multiscale(
        adata_combined.obsm[embedding_key], 
        perplexities=[50, 500], 
        metric="cosine", 
        n_jobs=n_jobs, 
        random_state=3
    )

    init = initialization.pca(adata_combined.obsm[embedding_key], random_state=42)

    embedding = TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
    )

    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

    adata_combined.obsm[output_key] = embedding
    
    print(f"✅ Standard t-SNE complete! Embeddings stored in obsm['{output_key}']")
    
    return adata_combined

#default method with perplexity based NN and PCA initialization
def compute_tsne_embedding(
    adata, 
    embedding_key="pca", 
    output_key="tsne_default",
    n_jobs=8
):
    """
    Computes t-SNE embeddings from a specified embedding in an AnnData object and stores the result in `obsm`.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the input embeddings in `obsm[embedding_key]`.
    embedding_key : str, optional
        The key in `adata.obsm` where the input embeddings are stored. Default is "pca".
    output_key : str, optional
        The key under which to store the computed t-SNE embeddings in `adata.obsm`. Default is "tsne_default".

    Returns
    -------
    AnnData
        The AnnData object with the t-SNE embeddings stored in `obsm[output_key]`.
    """
    print(f"Computing t-SNE embeddings with default method from {embedding_key} and storing in {output_key}...")
    print(f"Computing t-SNE embeddings from {embedding_key}...")
    print(f"Computing t-SNE for {adata.shape[1]} genes")
    affinities = openTSNE.affinity.PerplexityBasedNN(
        adata.obsm[embedding_key], 
        perplexity=30, 
        metric="cosine", 
        n_jobs=n_jobs, 
        random_state=3
    )

    init = initialization.pca(adata.obsm[embedding_key], random_state=0)

    embedding = TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
    )

    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

    # output_key = f"{output_key}_tsne"

    adata.obsm[output_key] = embedding
    return adata


