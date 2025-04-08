import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
    

def calculate_clustering_metrics(embedding, true_labels):
    """
    Calculate ARI and AMI metrics for embeddings by clustering them and comparing to true labels.
    
    Parameters:
    -----------
    embedding : numpy.ndarray
        The embedding coordinates (e.g., TSNE or UMAP)
    true_labels : numpy.ndarray or pandas.Series
        The true cell type labels
        
    Returns:
    --------
    dict
        Dictionary containing ARI and AMI scores
    """
   
    # Get the number of unique cell types
    n_clusters = len(np.unique(true_labels))
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    predicted_labels = kmeans.labels_
    
    # Ensure true_labels is a numpy array
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    
    return {"ARI": ari, "AMI": ami}