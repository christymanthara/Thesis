import anndata
import scanpy as sc

def print_adata_stats(file_path):
    """
    Reads an AnnData file and prints key statistics.

    Parameters:
    - file_path: str, path to the .h5ad file

    Outputs:
    - Shape of the dataset (Cells × Genes)
    - Number of genes before and after filtering
    - Summary of cell and gene metadata
    """
    
    # Load the AnnData object
    adata = anndata.read_h5ad(file_path)

    # Initial shape
    initial_genes = adata.shape[1]
    
    # Filter genes with at least one count
    sc.pp.filter_genes(adata, min_counts=1)
    filtered_genes = adata.shape[1]

    print(f"📊 Dataset Shape: {adata.shape} (Cells × Genes)")
    print(f"🧬 Genes Before Filtering: {initial_genes}")
    print(f"🧬 Genes After Filtering: {filtered_genes}")

    # Cell metadata summary
    print("\n🔍 Cell Metadata Overview (obs):")
    print(adata.obs.describe(include="all"))

    # Gene metadata summary
    print("\n🔬 Gene Metadata Overview (var):")
    print(adata.var.describe(include="all"))

    # Unique labels in obs (if available)
    if "labels" in adata.obs.columns:
        print("\n🏷️ Unique Labels in obs['labels']:")
        print(adata.obs["labels"].value_counts())

if __name__ == "__main__":
    # Example usage
    print_adata_stats("../datasets/baron_2016h.h5ad")
