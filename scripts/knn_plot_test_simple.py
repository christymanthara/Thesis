import anndata
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from .compute_tsne_embeddings import compute_tsne_embedding_pavlin
from .compute_tsne_embeddings import compute_tsne_embedding

from .knn_plot_table_simple import create_results_table

def compute_knn_tsne_simple(file_path, reference_file=None):
    """
    Simplified version: Compare how well different embeddings work for cell type classification
    
    What this does:
    1. Load your data
    2. Split by source if available, or use reference file
    3. For each embedding (PCA, UMAP, etc.), compute t-SNE
    4. Train a classifier on reference data
    5. Test how well it predicts cell types on new data
    6. Make plots and save results
    """
    results_table = {} 
    # Step 1: Load the data
    if isinstance(file_path, str):
        adata = anndata.read_h5ad(file_path)
    else:
        # Input is already an AnnData object
        adata = file_path

    print(f"📦 Available keys in uns (Unstructured Data):")
    print(list(adata.uns.keys()))
    
    # Step 2: Handle data splitting
    if reference_file:
        # Use separate reference file
        ref_adata = anndata.read_h5ad(reference_file)
        query_adata = adata.copy()
        print(f"Using separate reference file: {reference_file}")
    else:
        # Check if we can split by source
        if 'source' in adata.obs.columns:
            source_values = adata.obs['source'].unique()
            print(f"Found source values: {source_values}")
            
            if len(source_values) == 2:
                # Split into two datasets based on source
                source1, source2 = source_values
                print(f"Splitting data into {source1} and {source2}")
                print(f"Using {source1} as reference data and {source2} as query data")
                
                # Create separate AnnData objects for each source
                ref_adata = adata[adata.obs['source'] == source1].copy()
                query_adata = adata[adata.obs['source'] == source2].copy()
                
                print(f"Reference data shape: {ref_adata.shape}")
                print(f"Query data shape: {query_adata.shape}")
            else:
                print(f"Found {len(source_values)} source values, using same data as reference")
                ref_adata = adata.copy()
                query_adata = adata.copy()
        else:
            print("No 'source' column found, using same data as reference")
            ref_adata = adata.copy()
            query_adata = adata.copy()
    
    # Step 3: Find all embeddings in the data
    # Look for things like X_pca, X_umap, X_scvi, etc.
    
    print(f"📦 Available keys in ref adata uns (Unstructured Data):")
    print(list(ref_adata.uns.keys()))
    
    print(f"📦 Available keys in query adata uns (Unstructured Data):")
    print(list(query_adata.uns.keys()))
    
    
    embedding_keys = [key for key in query_adata.obsm.keys() if key.startswith('X_')]
    print(f"Found embeddings: {embedding_keys}")
    
    results = {}  # Store results for each embedding
    
    # Step 4: Process each embedding
    for embedding_name in embedding_keys:
        print(f"\n--- Processing {embedding_name} ---")
        
        # Skip if embedding doesn't exist in both datasets
        if embedding_name not in query_adata.obsm or embedding_name not in ref_adata.obsm:
            print(f"Skipping {embedding_name} - not found in both datasets")
            continue
        
        # Step 5: Compute t-SNE from this embedding using original data
        print(f"Computing t-SNE from {embedding_name}")
        adata_with_tsne = compute_tsne_for_embedding(adata, embedding_name)
        
        # Step 6: Prepare data for classification
        # Get reference data (training set)
        ref_embeddings = ref_adata.obsm[embedding_name]
        ref_labels = ref_adata.obs["labels"].values
        
        # Get query data (test set)  
        query_embeddings = query_adata.obsm[embedding_name]
        query_labels = query_adata.obs["labels"].values
        
        # Step 7: Train classifier and test
        print(f"Training KNN classifier on {embedding_name}")
        
        # Create and train classifier
        knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
        knn.fit(ref_embeddings, ref_labels)
        
        # Test on query data
        predictions = knn.predict(query_embeddings)
        accuracy = accuracy_score(query_labels, predictions)
        
        # Cross-validation on reference data
        cv_scores = cross_val_score(knn, ref_embeddings, ref_labels, cv=5)
        cv_accuracy = cv_scores.mean()
        
        print(f"Results for {embedding_name}:")
        print(f"  Cross-validation accuracy: {cv_accuracy:.3f}")
        print(f"  Transfer accuracy: {accuracy:.3f}")
        
        # Step 8: Test t-SNE version too
        tsne_key = f"{embedding_name}_tsne"
        tsne_accuracy = None
        if tsne_key in query_adata.obsm and tsne_key in ref_adata.obsm:
            knn_tsne = KNeighborsClassifier(n_neighbors=10)
            knn_tsne.fit(ref_adata.obsm[tsne_key], ref_labels)
            
            tsne_predictions = knn_tsne.predict(query_adata.obsm[tsne_key])
            tsne_accuracy = accuracy_score(query_labels, tsne_predictions)
            print(f"  t-SNE accuracy: {tsne_accuracy:.3f}")
        
        # Step 8: Store results
        results[embedding_name] = {
            'cv_accuracy': cv_accuracy,
            'transfer_accuracy': accuracy,
            'tsne_accuracy': tsne_accuracy
        }
        
        # Step 9: Make visualization
        embedding_key = embedding_name
        print(f"Creating t-SNE plot for {embedding_name}")
        if embedding_key == 'X_pca':
            # For PCA, create plots for both t-SNE methods
            tsne_key = f"{embedding_name}_tsne"
            if tsne_key in adata_with_tsne.obsm:
                make_plot(adata_with_tsne, tsne_key, f"{embedding_name}_default")
            
            multiscale_tsne_key = "X_pca_multiscale_tsne"
            if multiscale_tsne_key in adata_with_tsne.obsm:
                make_plot(adata_with_tsne, multiscale_tsne_key, f"{embedding_name}_multiscale")
        else:
            # For other embeddings, use default t-SNE
            tsne_key = f"{embedding_name}_tsne"
            if tsne_key in adata_with_tsne.obsm:
                make_plot(adata_with_tsne, tsne_key, embedding_name)
    
    # Step 10: Print summary
    print("\n=== SUMMARY ===")
    for embedding, metrics in results.items():
        print(f"{embedding}:")
        print(f"  CV: {metrics['cv_accuracy']:.3f}")
        print(f"  Transfer: {metrics['transfer_accuracy']:.3f}")
        if metrics['tsne_accuracy']:
            print(f"  t-SNE: {metrics['tsne_accuracy']:.3f}")
    # adding the data for the table creation        
    embedding_clean = embedding_key.replace('X_', '')

    # Special display name for original_X
    if results_table:
        # Get metadata (fix variable scope issues)
        source1 = source_values[0] if 'source_values' in locals() and len(source_values) >= 1 else "unknown"
        source2 = source_values[1] if 'source_values' in locals() and len(source_values) >= 2 else "unknown"
        
        ref_tissue = ref_adata.uns.get('tissue', 'unknown')
        query_tissue = query_adata.uns.get('tissue', 'unknown')
        
        ref_organism = ref_adata.uns.get('organism', 'unknown')
        query_organism = query_adata.uns.get('organism', 'unknown')
        
        ref_cell_count = ref_adata.n_obs
        query_cell_count = query_adata.n_obs
        
        metadata = {
            'ref_source': source1,
            'query_source': source2,
            'ref_tissue': ref_tissue,
            'query_tissue': query_tissue,
            'ref_organism': ref_organism,
            'query_organism': query_organism,
            'ref_cell_count': ref_cell_count,
            'query_cell_count': query_cell_count
        }
        
        base_filename = f"results_table_{source1}_{source2}"    
        create_results_table(results_table, source1, source2, base_filename, metadata=metadata)
        print("Results table created successfully.")
        print(f"\nCompleted processing all embeddings for {file_path}")

    return results


def compute_tsne_for_embedding(adata, embedding_key):
    """
    Simple wrapper to compute t-SNE from an embedding
    """
    adata_copy = adata.copy()
    
    # Skip if it's already UMAP (2D visualization)
    if 'umap' in embedding_key.lower():
        return adata_copy
    
    # Use your existing t-SNE functions
    if embedding_key == 'X_pca':
        # Use both methods for PCA
        # First use Pavlin's method
        adata_copy = compute_tsne_embedding_pavlin(adata_copy, embedding_key, output_key="X_pca_multiscale_tsne")
        # Then use default method
        adata_copy = compute_tsne_embedding(adata_copy, embedding_key, output_key=f"{embedding_key}_tsne")
        return adata_copy
    else:
        # Use default method for others
        return compute_tsne_embedding(adata_copy, embedding_key, output_key=f"{embedding_key}_tsne")


def make_plot(adata_with_tsne, tsne_key, embedding_name):
    """
    Simple plotting function - splits t-SNE embeddings by source for visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get the t-SNE coordinates and labels for all data
    tsne_coords = adata_with_tsne.obsm[tsne_key]
    labels = adata_with_tsne.obs["labels"]
    sources = adata_with_tsne.obs["source"]
    
    # Get unique sources and labels
    unique_sources = sources.unique()
    unique_labels = labels.unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot source 1 (reference) - left subplot
    source1_mask = sources == unique_sources[0]
    source1_coords = tsne_coords[source1_mask]
    source1_labels = labels[source1_mask]
    
    for i, label in enumerate(unique_labels):
        label_mask = source1_labels == label
        if label_mask.any():
            ax1.scatter(source1_coords[label_mask, 0], source1_coords[label_mask, 1], 
                       c=[colors[i]], label=label, s=3, alpha=0.7)
    
    ax1.set_title(f"Reference - {unique_sources[0]} ({embedding_name})")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axis('off')
    
    # Plot source 2 (query) - right subplot
    if len(unique_sources) > 1:
        source2_mask = sources == unique_sources[1]
        source2_coords = tsne_coords[source2_mask]
        source2_labels = labels[source2_mask]
        
        
        for i, label in enumerate(unique_labels):
            label_mask = source1_labels == label
            if label_mask.any():
                ax2.scatter(source1_coords[label_mask, 0], source1_coords[label_mask, 1], 
                           c='lightgrey',  s=3, alpha=0.7)
        
        for i, label in enumerate(unique_labels):
            label_mask = source2_labels == label
            if label_mask.any():
                ax2.scatter(source2_coords[label_mask, 0], source2_coords[label_mask, 1], 
                           c=[colors[i]], label=label, s=3, alpha=0.7)

            
        ax2.set_title(f"Query - {unique_sources[1]} ({embedding_name})")
        ax2.axis('off')
    else:
        # If only one source, show same data in both plots
        for i, label in enumerate(unique_labels):
            label_mask = source1_labels == label
            if label_mask.any():
                ax2.scatter(source1_coords[label_mask, 0], source1_coords[label_mask, 1], 
                           c=[colors[i]], label=label, s=3, alpha=0.7)
        ax2.set_title(f"Same data ({embedding_name})")
    
    plt.tight_layout()
    plt.savefig(f"plot_{embedding_name}.pdf", bbox_inches='tight')
    plt.close()  # Close the figure to free memory


# Example usage:
# results = compute_knn_tsne_simple("my_data.h5ad")  # Will split by source automatically
# results = compute_knn_tsne_simple("my_data.h5ad", "reference_data.h5ad")  # Use separate reference