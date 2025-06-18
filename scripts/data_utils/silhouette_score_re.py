import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import pandas as pd

def compute_batch_effect_metrics(embeddings, cell_types, batch_labels, 
                               embedding_name, k_neighbors=30):
    """
    Compute metrics to evaluate batch effect reduction.
    
    For good batch integration, we want:
    1. Low batch silhouette score (batches should be mixed, not separated)
    2. High cell type silhouette score (cell types should remain separated)
    3. High batch mixing entropy (neighbors should come from different batches)
    4. Low average silhouette width for batches within each cell type
    
    Parameters:
    -----------
    embeddings : array-like, shape (n_cells, n_features)
        Cell embeddings
    cell_types : array-like, shape (n_cells,)
        Cell type labels
    batch_labels : array-like, shape (n_cells,)
        Batch/source labels (e.g., "baron_2016h", "xin_2016")
    embedding_name : str
        Name of embedding for logging
    k_neighbors : int, default=30
        Number of neighbors to consider for mixing metrics
        
    Returns:
    --------
    dict
        Dictionary containing batch effect metrics:
        - 'batch_silhouette': Silhouette score using batch labels (lower is better)
        - 'celltype_silhouette': Silhouette score using cell type labels (higher is better)
        - 'batch_mixing_entropy': Average entropy of batch mixing (higher is better)
        - 'asw_batch': Average silhouette width for batches within cell types (lower is better)
        - 'integration_score': Combined score (higher is better)
    """
    
    metrics = {
        'batch_silhouette': 0.0,
        'celltype_silhouette': 0.0, 
        'batch_mixing_entropy': 0.0,
        'asw_batch': 0.0,
        'integration_score': 0.0
    }
    
    try:
        print(f"Computing batch effect metrics for {embedding_name}...")
        
        # Convert to numpy arrays and ensure consistent types
        embeddings = np.array(embeddings)
        cell_types = np.array(cell_types).astype(str)
        batch_labels = np.array(batch_labels).astype(str)
        
        # Check if we have sufficient data
        unique_batches = len(set(batch_labels))
        unique_celltypes = len(set(cell_types))
        n_cells = len(embeddings)
        
        if n_cells < 2 or unique_batches < 2:
            print(f"Warning: Insufficient data for batch metrics (cells: {n_cells}, batches: {unique_batches})")
            return metrics
            
        # 1. Batch Silhouette Score (lower is better - we want batches to be mixed)
        if unique_batches > 1:
            batch_sil = silhouette_score(embeddings, batch_labels)
            metrics['batch_silhouette'] = batch_sil
            print(f"Batch silhouette score: {batch_sil:.4f} (lower is better)")
        
        # 2. Cell Type Silhouette Score (higher is better - we want cell types separated)
        if unique_celltypes > 1:
            celltype_sil = silhouette_score(embeddings, cell_types)
            metrics['celltype_silhouette'] = celltype_sil
            print(f"Cell type silhouette score: {celltype_sil:.4f} (higher is better)")
        
        # 3. Batch Mixing Entropy
        # For each cell, check what batches its k nearest neighbors come from
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_cells)).fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)
        
        mixing_entropies = []
        for i, neighbors in enumerate(indices):
            # Exclude self (first neighbor)
            neighbor_batches = batch_labels[neighbors[1:]]
            # Count batch occurrences
            batch_counts = Counter(neighbor_batches)
            # Calculate entropy
            total = len(neighbor_batches)
            entropy = -sum((count/total) * np.log2(count/total) 
                          for count in batch_counts.values() if count > 0)
            mixing_entropies.append(entropy)
        
        avg_mixing_entropy = np.mean(mixing_entropies)
        metrics['batch_mixing_entropy'] = avg_mixing_entropy
        print(f"Batch mixing entropy: {avg_mixing_entropy:.4f} (higher is better)")
        
        # 4. Average Silhouette Width for batches within each cell type
        # This measures how well batches are mixed within each cell type
        asw_scores = []
        for celltype in set(cell_types):
            celltype_mask = cell_types == celltype
            celltype_embeddings = embeddings[celltype_mask]
            celltype_batches = batch_labels[celltype_mask]
            
            # Skip if only one batch or too few cells in this cell type
            if len(set(celltype_batches)) > 1 and len(celltype_embeddings) > 1:
                try:
                    asw = silhouette_score(celltype_embeddings, celltype_batches)
                    asw_scores.append(asw)
                except:
                    continue
        
        if asw_scores:
            avg_asw = np.mean(asw_scores)
            metrics['asw_batch'] = avg_asw
            print(f"Average silhouette width (batch within cell type): {avg_asw:.4f} (lower is better)")
        
        # 5. Integration Score (combined metric)
        # Higher cell type separation, lower batch separation, higher mixing
        max_entropy = np.log2(unique_batches)  # Maximum possible entropy
        normalized_entropy = avg_mixing_entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine metrics (scale to 0-1 range)
        celltype_component = max(0, (celltype_sil + 1) / 2)  # Scale from [-1,1] to [0,1]
        batch_component = max(0, (1 - batch_sil) / 2)  # Invert and scale
        entropy_component = normalized_entropy
        asw_component = max(0, (1 - (avg_asw if asw_scores else 0)) / 2)  # Invert and scale
        
        integration_score = np.mean([celltype_component, batch_component, 
                                   entropy_component, asw_component])
        metrics['integration_score'] = integration_score
        print(f"Integration score: {integration_score:.4f} (higher is better)")
        
    except Exception as e:
        print(f"Error computing batch effect metrics for {embedding_name}: {str(e)}")
    
    return metrics


def compute_batch_integration_comparison(reference_embeddings, reference_celltypes, reference_batches,
                                       query_embeddings, query_celltypes, query_batches,
                                       combined_embeddings, combined_celltypes, combined_batches,
                                       embedding_name):
    """
    Compare batch integration across reference, query, and combined datasets.
    
    Parameters:
    -----------
    reference_embeddings, query_embeddings, combined_embeddings : array-like
        Embeddings for each dataset
    reference_celltypes, query_celltypes, combined_celltypes : array-like  
        Cell type labels for each dataset
    reference_batches, query_batches, combined_batches : array-like
        Batch labels for each dataset
    embedding_name : str
        Name of embedding method
        
    Returns:
    --------
    dict
        Dictionary with metrics for each dataset
    """
    
    results = {}
    
    # Individual dataset metrics
    results['reference'] = compute_batch_effect_metrics(
        reference_embeddings, reference_celltypes, reference_batches, 
        f"{embedding_name}_reference"
    )
    
    results['query'] = compute_batch_effect_metrics(
        query_embeddings, query_celltypes, query_batches,
        f"{embedding_name}_query" 
    )
    
    # Combined dataset metrics (most important for batch effect evaluation)
    results['combined'] = compute_batch_effect_metrics(
        combined_embeddings, combined_celltypes, combined_batches,
        f"{embedding_name}_combined"
    )
    
    return results