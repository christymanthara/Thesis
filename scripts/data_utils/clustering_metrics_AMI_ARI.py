import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans

def calculate_clustering_metrics(embedding, true_labels):
    """
    Calculate ARI and AMI metrics for embeddings by clustering them and comparing to true labels.
    
    Parameters:
    -----------
    embedding : numpy.ndarray
        The embedding coordinates (e.g., TSNE or UMAP), shape (n_samples, n_features)
    true_labels : array-like
        The true cell type labels (Pandas Series or array)
        
    Returns:
    --------
    dict
        Dictionary containing ARI and AMI scores
    """
    # Ensure embedding is 2D
    embedding = np.asarray(embedding)
    assert embedding.ndim == 2, "Embedding must be a 2D array."

    # Convert true_labels safely
    if hasattr(true_labels, 'to_numpy'):
        true_labels = true_labels.to_numpy()
    else:
        true_labels = np.asarray(true_labels)
    
    # Get number of unique classes from the ground truth
    n_clusters = len(np.unique(true_labels))
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')  # use n_init=10 if older sklearn
    predicted_labels = kmeans.fit_predict(embedding)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    
    return {"ARI": ari, "AMI": ami}
