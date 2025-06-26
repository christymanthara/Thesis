import anndata
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd

def transform_tsne_single(adata_path: str, new_path: str, results_table=None):
    """
    Transform new data onto existing t-SNE embedding using all shared genes.
    
    Parameters:
    -----------
    adata_path : str
        Path to reference AnnData file (should have X_pavlin_tsne embedding)
    new_path : str  
        Path to new AnnData file to transform
    results_table : dict, optional
        Existing results table to append results to
    
    Returns:
    --------
    tuple: (transformed_data, updated_results_table)
    """
    # Initialize results_table if not provided
    if results_table is None:
        results_table = {}
    
    
    def extract_filename(path):
        filename = os.path.basename(path)  # Get file name
        return filename.rsplit('.h5ad', 1)[0]  # Remove the extension


    
    
    
    # Load datasets
    if isinstance(adata_path, str):
        adata = anndata.read_h5ad(adata_path)
        print(f"Loaded AnnData object from file path")
        file_path = adata_path
        adata_name = extract_filename(adata_path)

        # Add source labels 
        adata.obs["source"] = pd.Categorical([adata_name] * adata.n_obs)
        
        
    else:
        print("Using provided AnnData object for reference data")
        adata = adata_path
        sources = adata.obs["source"]
        adata_name = sources.unique()[0] if isinstance(sources, pd.Series) else "reference"
        print(f"Using source label: {adata_name}")
        
    if isinstance(new_path, str):
        print(f"Loading new data from {new_path}")
        new = anndata.read_h5ad(new_path)
        file_path = new_path
        new_name = extract_filename(new_path)
        print(f"New data name: {new_name}")

        # FIXED: Using new_name and new.n_obs
        new.obs["source"] = pd.Categorical([new_name] * new.n_obs)
    else:
        print("Using provided AnnData object for new data")
        new = new_path
        # You should also handle the case where new_path is an AnnData object
        # and might not have source labels yet
        if "source" not in new.obs.columns:
            new_name = "query_data"  # or extract from existing source if available
            new.obs["source"] = pd.Categorical([new_name] * new.n_obs)
        
    

    
    
    print("Original labels in new (before filtering):", new.obs["labels"].value_counts())
    print("Original labels in adata (before filtering):", adata.obs["labels"].value_counts())
    
    # Find shared genes and filter both datasets
    shared_genes = adata.var_names[adata.var_names.isin(new.var_names)]
    adata = adata[:, adata.var_names.isin(shared_genes)]
    new = new[:, new.var_names.isin(shared_genes)]
    
    # Sort genes to ensure same order
    adata = adata[:, adata.var_names.argsort()].copy()
    new = new[:, new.var_names.argsort()].copy()
    assert all(adata.var_names == new.var_names)
    
    print(f"Using {adata.shape[1]} shared genes")
    print(f"Reference data shape: {adata.shape}")
    print(f"New data shape: {new.shape}")
    
    # Check if reference has t-SNE embedding
    if "X_pavlin_tsne" not in adata.obsm:
        raise ValueError("Reference data must have 'X_pavlin_tsne' embedding. Run preprocessing pipeline first.")
    
    print(f"Reference t-SNE shape: {adata.obsm['X_pavlin_tsne'].shape}")
    
    # Compute affinities for transformation
    print("Computing affinities for transformation...")
    affinities = affinity.PerplexityBasedNN(
        adata.X.toarray() if sp.issparse(adata.X) else adata.X,
        perplexity=30,
        metric="cosine",
        n_jobs=8,
        random_state=3,
    )
    
    # Create embedding object from reference
    embedding = TSNEEmbedding(
        adata.obsm["X_pavlin_tsne"],
        affinities,
        negative_gradient_method="fft",
        n_jobs=8,
    )
    
    # Transform new data
    print("Transforming new data...")
    new_embedding = embedding.prepare_partial(
        new.X.toarray() if sp.issparse(new.X) else new.X, 
        k=10
    )
    new.obsm["tsne_init"] = new_embedding.copy()
    
    # Optimize new embedding
    new_embedding.optimize(250, learning_rate=0.1, momentum=0.8, inplace=True)
    
    # Convert embeddings to numpy arrays to ensure HDF5 compatibility
    new.obsm["tsne_init"] = np.array(new.obsm["tsne_init"])
    new.obsm["X_pavlin_tsne"] = np.array(new_embedding)
    
    print(f"New data t-SNE shape: {new.obsm['X_pavlin_tsne'].shape}")
    
    # Train KNN classifier on reference data
    embedding_name = "X_pavlin_tsne"
    ref_adata = adata
    query_adata = new
    ref_embeddings = ref_adata.obsm[embedding_name]
    ref_labels = ref_adata.obs["labels"].values
    query_embeddings = query_adata.obsm[embedding_name]
    query_labels = query_adata.obs["labels"].values
        
    knn = KNeighborsClassifier()
    knn.fit(ref_embeddings, ref_labels)
    # Predict labels for new data
    predictions = knn.predict(query_embeddings)
    accuracy = accuracy_score(query_labels, predictions)
    
    cv_scores = cross_val_score(knn, ref_embeddings, ref_labels, cv=5)
    cv_accuracy = cv_scores.mean()
    
    print(f"Results for {embedding_name}:")
    print(f"  Cross-validation accuracy: {cv_accuracy:.3f}")
    print(f"  Transfer accuracy: {accuracy:.3f}")
    
    # ADD RESULTS TO TABLE (same format as first function)
    display_name = "Transformed t-SNE"  # You can customize this name
    results_table[display_name] = {
        'Reference CV': f"{cv_accuracy:.3f}",
        'Query Transfer': f"{accuracy:.3f}",
    }

    # Calculate clustering metrics if available
    try:
        from data_utils import clustering_metrics_AMI_ARI
        embedding_metrics = clustering_metrics_AMI_ARI.calculate_clustering_metrics(
            new.obsm["X_pavlin_tsne"], 
            new.obs["labels"]
        )
        
        print(f"Clustering metrics:")
        print(f"  ARI: {embedding_metrics['ARI']:.4f}")
        print(f"  AMI: {embedding_metrics['AMI']:.4f}")
        
        # Save metrics (convert to native Python types for HDF5 compatibility)
        new.uns["ari"] = float(embedding_metrics["ARI"])
        new.uns["ami"] = float(embedding_metrics["AMI"])
        
    except ImportError:
        print("clustering_metrics_AMI_ARI not available, skipping metrics calculation")
        embedding_metrics = None
    
    # Save transformed data
    output_filename = f"transformed_{embedding_name}_{adata_name}_{new_name}.h5ad"
    new.write_h5ad(output_filename)
    print(f"Transformed data saved as: {output_filename}")
    
    # Create visualization
    create_transformation_plot(adata, new, adata_name, new_name, embedding_metrics)
    
    return new, results_table

def create_transformation_plot(adata, new, adata_path, new_path, metrics=None):
    """
    Create visualization comparing reference and transformed data.
    
    Parameters:
    -----------
    adata : AnnData
        Reference data
    new : AnnData  
        Transformed new data
    adata_path : str
        Path to reference data (for filename)
    new_path : str
        Path to new data (for filename)  
    metrics : dict, optional
        Clustering metrics dictionary
    """
    try:
        import utils
        colors = utils.get_colors_for(adata)
    except:
        # Fallback color scheme
        unique_labels = np.unique(adata.obs["labels"])
        colors = {label: plt.cm.tab10(i) for i, label in enumerate(unique_labels)}
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot reference data
    for label in np.unique(adata.obs["labels"]):
        mask = adata.obs["labels"] == label
        ax1.scatter(
            adata.obsm["X_pavlin_tsne"][mask, 0], 
            adata.obsm["X_pavlin_tsne"][mask, 1],
            c=[colors.get(label, 'gray')], 
            label=label, 
            s=3, 
            alpha=0.7
        )
    ax1.set_title("Reference Data")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot transformed data on reference background
    # First plot reference as background
    for label in np.unique(adata.obs["labels"]):
        mask = adata.obs["labels"] == label
        ax2.scatter(
            adata.obsm["X_pavlin_tsne"][mask, 0], 
            adata.obsm["X_pavlin_tsne"][mask, 1],
            c=[colors.get(label, 'gray')], 
            s=1, 
            alpha=0.1
        )
    
    # Then plot transformed data on top
    for label in np.unique(new.obs["labels"]):
        mask = new.obs["labels"] == label
        ax2.scatter(
            new.obsm["X_pavlin_tsne"][mask, 0], 
            new.obsm["X_pavlin_tsne"][mask, 1],
            c=[colors.get(label, 'gray')], 
            label=label, 
            s=12, 
            alpha=1.0,
            edgecolors='black',
            linewidth=0.1
        )
    
    # Add metrics to title if available
    title = "Transformed Data"
    if metrics:
        title += f" (ARI: {metrics['ARI']:.3f}, AMI: {metrics['AMI']:.3f})"
    ax2.set_title(title)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    # filename1 = os.path.splitext(os.path.basename(adata_path))[0]
    # filename2 = os.path.splitext(os.path.basename(new_path))[0]
    output_pdf = f"{adata_path}-{new_path}-transformation.pdf"
    
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Plot saved as {output_pdf}")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Transform new data onto reference embedding
    transformed_data = transform_tsne_single(
        "F:/Thesis/baron_2016h_embedding_tsne_all_genes.h5ad", 
        "F:/Thesis/Datasets/xin_2016.h5ad"
    )