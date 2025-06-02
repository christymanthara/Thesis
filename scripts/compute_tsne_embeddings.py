from openTSNE import TSNEEmbedding, affinity, initialization


def compute_tsne_embedding(
    adata, 
    embedding_key="X_scVI", 
    output_key="test"
):
    """
    Computes t-SNE embeddings from a specified embedding in an AnnData object and stores the result in `obsm`.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the input embeddings in `obsm[embedding_key]`.
    embedding_key : str, optional
        The key in `adata.obsm` where the input embeddings are stored. Default is "X_scVI".
    output_key : str, optional
        The key under which to store the computed t-SNE embeddings in `adata.obsm`. Default is "tsne_scvi".

    Returns
    -------
    AnnData
        The AnnData object with the t-SNE embeddings stored in `obsm[output_key]`.
    """
    print(f"Computing t-SNE embeddings from {embedding_key}...")
    print(f"Computing t-SNE for {adata.shape[1]} genes")
    affinities = affinity.Multiscale(
        adata.obsm[embedding_key], 
        perplexities=[30, 100], 
        metric="cosine", 
        n_jobs=8, 
        random_state=3
    )

    init = initialization.pca(adata.obsm[embedding_key], random_state=42)

    embedding = TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=8,
    )

    embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

    # output_key = f"{output_key}_tsne"

    adata.obsm[output_key] = embedding
    return adata