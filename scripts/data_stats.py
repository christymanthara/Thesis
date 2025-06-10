import anndata
import scanpy as sc
import numpy as np


def analyze_adata(adata_path, title=None,  cell_type_column="", batch_column=""):
    """
    Analyze and print detailed statistics about an AnnData object,
    including inspection of gene identifiers and obsm values.
    
    Parameters:
    -----------
    adata_path : str
        Path to the AnnData h5ad file
    title : str, optional
        Optional title to display before the analysis
    cell_type_column : str, optional
        Column name in obs containing cell type information
    batch_column : str, optional
        Column name in obs containing batch information
    """

    adata = anndata.read_h5ad(adata_path)
    if title:
        print(f"\n{'='*50}\n{title}\n{'='*50}")
    
    # Basic dataset information
    print(f"📊 Dataset Shape: {adata.shape} (Cells × Genes)")
    
    # Gene filtering information if available in uns
    if 'genes_before_filter' in adata.uns:
        print(f"🧬 Genes Before Filtering: {adata.uns['genes_before_filter']}")
    print(f"🧬 Genes After Filtering: {adata.n_vars}")
    
    # Metadata in obs
    print(f"🔍 Available columns in obs (Cell Metadata):")
    print(list(adata.obs.columns))
    
    # Show unique values in each obs column
    print(f"\n📋 Unique values in each obs column:")
    for col in adata.obs.columns:
        unique_vals = adata.obs[col].unique()
        n_unique = len(unique_vals)
        
        # For columns with few unique values, show all values
        if n_unique <= 10:
            print(f"  🔹 {col} ({n_unique} unique): {sorted(unique_vals.tolist())}")
        else:
            # For columns with many unique values, show count and first few examples
            print(f"  🔹 {col} ({n_unique} unique): {sorted(unique_vals.tolist())[:5]}... (showing first 5)")
        
        # Show value counts for categorical-like data
        if n_unique <= 20:  # Only show counts for columns with reasonable number of categories
            value_counts = adata.obs[col].value_counts()
            print(f"      Value counts: {dict(value_counts)}")
    
    
    # Metadata in var
    print(f"🔬 Available columns in var (Gene Metadata):")
    print(list(adata.var.columns))
    
    # Inspect gene identifiers
    print(f"🔬 Gene Identifiers (first 5):")
    print(adata.var.index[:5].tolist())
    
    # Check for gene_name or similar columns
    gene_id_columns = [col for col in adata.var.columns if 'gene' in col.lower() or 'symbol' in col.lower()]
    if gene_id_columns:
        print(f"🧬 Potential gene identifier columns: {gene_id_columns}")
        for col in gene_id_columns:
            print(f"  - Column '{col}' (first 5 values): {adata.var[col][:5].tolist()}")
    else:
        print("🧬 No explicit gene name/symbol columns found. Gene identifiers are likely in the var index.")
    
    
    # Cell type and batch analysis
    print(f"\n📋 Cell Type and Batch Analysis:")
    if not cell_type_column and not batch_column:
        print("  Parameters not passed.")
    else:
        if cell_type_column:
            if cell_type_column in adata.obs.columns:
                unique_cell_types = len(adata.obs[cell_type_column].unique())
                print(f"  🧬 Number of unique cell types in '{cell_type_column}': {unique_cell_types}")
            else:
                print(f"  ⚠️ Cell type column '{cell_type_column}' not found in obs.")
        
        if batch_column:
            if batch_column in adata.obs.columns:
                unique_batches = len(adata.obs[batch_column].unique())
                batch_names = adata.obs[batch_column].unique().tolist()
                print(f"  🔬 Number of unique batches in '{batch_column}': {unique_batches}")
                print(f"  🔬 Batch names: {batch_names}")
            else:
                print(f"  ⚠️ Batch column '{batch_column}' not found in obs.")
    
    # Labels if available
    if 'labels' in adata.obs:
        print(f"🏷️ Unique Labels in obs['labels']:")
        print(adata.obs['labels'].value_counts())
    
    # Available unstructured data
    print(f"📦 Available keys in uns (Unstructured Data):")
    print(list(adata.uns.keys()))
    
    # Show obsm values if they exist
    if hasattr(adata, 'obsm') and adata.obsm:
        print(f"\n📐 Available keys in obsm (Multi-dimensional Annotations):")
        for key in adata.obsm.keys():
            shape = adata.obsm[key].shape
            print(f"  - {key}: shape {shape}")
            # Show a preview of the first few values for each embedding
            if isinstance(adata.obsm[key], np.ndarray):
                print(f"    Preview (first 2 cells, up to 5 dimensions):")
                preview = adata.obsm[key][:2, :min(5, adata.obsm[key].shape[1])]
                for i, row in enumerate(preview):
                    print(f"      Cell {i}: {row}")
            else:
                print(f"    Type: {type(adata.obsm[key])}")
    else:
        print("\n📐 No obsm (Multi-dimensional Annotations) available in this dataset.")


if __name__ == "__main__":
    # Example usage
    # print_adata_stats("extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad")
    # print_adata_stats("extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad")
    # print_adata_stats("../datasets/baron_2016h.h5ad")
    # print_adata_stats("/home/thechristyjo/Documents/Thesis/datasets/baron_2016h.h5ad")
    # print_adata_stats("/home/thechristyjo/Documents/Thesis/datasets/xin_2016.h5ad")
    # analyze_adata(
    #     "/home/thechristyjo/Documents/Thesis/datasets/baron_2016h.h5ad",
    #     title="Baron 2016 Dataset Analysis"
    # )
    # analyze_adata(
    #     "/home/thechristyjo/Documents/Thesis/datasets/xin_2016.h5ad",
    #     title="Xin 2016 Dataset Analysis"
    # )
    analyze_adata(
        "adata_concat_scGPT_baron_2016h_xin_2016.h5ad",
        title="Concatenated scGPT Dataset Analysis"
    )
    # analyze_adata(
    #     "F:/Thesis/UCE/baron_2016h_uce_adata.h5ad",
    #     title="Baron UCE"
    # )
    analyze_adata(
        "F:/Thesis/Datasets/baron_2016h_uce_adata.h5ad",
        title="Baron UCE"
    )
    
    analyze_adata(
        "F:/Thesis/adata_concat_uce_baron_2016h_uce_adata_xin_2016_uce_adata.h5ad",
        title="Concatenated UCE Baron and Xin Datasets with uce_adata"
    )
    
    analyze_adata(
        "adata_concat_scGPT_baron_2016h_xin_2016.h5ad",
        title="Concatenated scGPT Baron and Xin Datasets",
        batch_column="batch_id",
        cell_type_column="labels"
    )
    
    analyze_adata(
        "F:/Thesis/adata_concat_scGPT_baron_2016h_xin_2016_X_scvi_X_scanvi_X_scGPT.h5ad",
        title="Concatenated scGPT Baron and Xin Datasets with embeddings",
        batch_column="batch_id",
        cell_type_column="labels"
    )
    
    analyze_adata(
        "F:/Thesis/baron_2016hxin_2016.h5ad",
        title="Baron 2016h and Xin 2016 Dataset Analysis",
        batch_column="batch_id",    
        cell_type_column="labels"
    ) #the purest dataset with no embeddings, just the raw data
    # analyze_adata(
    #     "F:/Thesis/adata_concat_UCE_sample_proc_lung_train_uce_adata_sample_proc_lung_test_uce_adata.h5ad",
    #     title="lung data"
    # )
    
    
    analyze_adata(
        "F:/Thesis/baron_2016hxin_2016_uce_adata.h5ad",
        title="Baron 2016h and Xin 2016 Dataset Analysis with UCE",
        batch_column="batch_id",    
        cell_type_column="labels"
    )
    
    analyze_adata(
        "baron_2016hxin_2016_uce_adata_X_scvi_X_scanvi_X_scGPT_test.h5ad",
        title="Baron 2016h and Xin 2016 Dataset Analysis with UCE and other embeddings",
        batch_column="batch_id",    
        cell_type_column="labels"
    )
        