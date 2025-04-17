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
    - Lists available columns in obs (cell metadata) and var (gene metadata)
    - Prints available keys in uns (unstructured data)
    """
    
    # Load the AnnData object
    adata = anndata.read_h5ad(file_path)

    # Initial shape
    initial_genes = adata.shape[1]
    
    # Filter genes with at least one count
    sc.pp.filter_genes(adata, min_counts=1)
    filtered_genes = adata.shape[1]

    print(f"\n📊 Dataset Shape: {adata.shape} (Cells × Genes)")
    print(f"🧬 Genes Before Filtering: {initial_genes}")
    print(f"🧬 Genes After Filtering: {filtered_genes}")

    # Cell metadata summary
    print("\n🔍 Available columns in obs (Cell Metadata):")
    if not adata.obs.empty:
        print(list(adata.obs.columns))
    else:
        print("⚠️ No columns found in obs.")

    # Gene metadata summary
    print("\n🔬 Available columns in var (Gene Metadata):")
    if not adata.var.empty:
        print(list(adata.var.columns))
    else:
        print("⚠️ No columns found in var.")

    # Unique labels in obs (if available)
    if "labels" in adata.obs.columns:
        print("\n🏷️ Unique Labels in obs['labels']:")
        print(adata.obs["labels"].value_counts())

    # Unstructured metadata
    print("\n📦 Available keys in uns (Unstructured Data):")
    if adata.uns:
        print(list(adata.uns.keys()))
    else:
        print("⚠️ No keys found in uns.")

if __name__ == "__main__":
    # Example usage
    print_adata_stats("extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad")
    print_adata_stats("extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad")
    # print_adata_stats("../datasets/baron_2016h.h5ad")
