from sklearn.metrics import silhouette_score

def compute_silhouette_scores(reference_embeddings, reference_labels, 
                            query_embeddings, query_labels,
                            reference_tsne_embeddings, query_tsne_embeddings,
                            embedding_key, tsne_key):
    """
    Compute silhouette scores for both original and t-SNE embeddings.
    
    Parameters:
    -----------
    reference_embeddings : array-like
        Reference dataset embeddings (original space)
    reference_labels : array-like
        Reference dataset cell type labels
    query_embeddings : array-like
        Query dataset embeddings (original space)
    query_labels : array-like
        Query dataset cell type labels
    reference_tsne_embeddings : array-like
        Reference dataset t-SNE embeddings
    query_tsne_embeddings : array-like
        Query dataset t-SNE embeddings
    embedding_key : str
        Name of the original embedding (for logging)
    tsne_key : str
        Name of the t-SNE embedding (for logging)
    
    Returns:
    --------
    dict
        Dictionary containing all silhouette scores with keys:
        - 'reference_sil_score'
        - 'query_sil_score' 
        - 'reference_tsne_sil_score'
        - 'query_tsne_sil_score'
    """
    
    silhouette_results = {
        'reference_sil_score': 0.0,
        'query_sil_score': 0.0,
        'reference_tsne_sil_score': 0.0,
        'query_tsne_sil_score': 0.0
    }
    
    try:
        # Calculate silhouette score for original embeddings
        print(f"Computing silhouette score for {embedding_key}...")
        
        # For reference data - check if we have enough samples and clusters
        unique_ref_labels = len(set(reference_labels))
        if len(reference_embeddings) > 1 and unique_ref_labels > 1:
            reference_sil_score = silhouette_score(reference_embeddings, reference_labels)
            silhouette_results['reference_sil_score'] = reference_sil_score
            print(f"Reference silhouette score using {embedding_key}: {reference_sil_score:.4f}")
        else:
            print(f"Warning: Insufficient data for reference silhouette score (samples: {len(reference_embeddings)}, labels: {unique_ref_labels})")
        
        # For query data
        unique_query_labels = len(set(query_labels))
        if len(query_embeddings) > 1 and unique_query_labels > 1:
            query_sil_score = silhouette_score(query_embeddings, query_labels)
            silhouette_results['query_sil_score'] = query_sil_score
            print(f"Query silhouette score using {embedding_key}: {query_sil_score:.4f}")
        else:
            print(f"Warning: Insufficient data for query silhouette score (samples: {len(query_embeddings)}, labels: {unique_query_labels})")
        
        # Calculate silhouette score for t-SNE embeddings
        print(f"Computing silhouette score for {tsne_key}...")
        
        # For reference t-SNE data
        if len(reference_tsne_embeddings) > 1 and unique_ref_labels > 1:
            reference_tsne_sil_score = silhouette_score(reference_tsne_embeddings, reference_labels)
            silhouette_results['reference_tsne_sil_score'] = reference_tsne_sil_score
            print(f"Reference silhouette score using {tsne_key}: {reference_tsne_sil_score:.4f}")
        else:
            print(f"Warning: Insufficient data for reference t-SNE silhouette score")
        
        # For query t-SNE data
        if len(query_tsne_embeddings) > 1 and unique_query_labels > 1:
            query_tsne_sil_score = silhouette_score(query_tsne_embeddings, query_labels)
            silhouette_results['query_tsne_sil_score'] = query_tsne_sil_score
            print(f"Query silhouette score using {tsne_key}: {query_tsne_sil_score:.4f}")
        else:
            print(f"Warning: Insufficient data for query t-SNE silhouette score")
            
    except Exception as e:
        print(f"Error computing silhouette scores for {embedding_key}: {str(e)}")
        # Return default values if computation fails
        
    return silhouette_results