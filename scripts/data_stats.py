import anndata
import scanpy as sc


def analyze_adata(adata_path, title=None):
    """
    Analyze and print detailed statistics about an AnnData object,
    including inspection of gene identifiers.
    
    Parameters:
    -----------
    adata : AnnData
        The AnnData object to analyze
    title : str, optional
        Optional title to display before the analysis
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
    
    # Labels if available
    if 'labels' in adata.obs:
        print(f"🏷️ Unique Labels in obs['labels']:")
        print(adata.obs['labels'].value_counts())
    
    # Available unstructured data
    print(f"📦 Available keys in uns (Unstructured Data):")
    print(list(adata.uns.keys()))


# def print_adata_stats(file_path):
#     """
#     Reads an AnnData file and prints key statistics.

#     Parameters:
#     - file_path: str, path to the .h5ad file

#     Outputs:
#     - Shape of the dataset (Cells × Genes)
#     - Number of genes before and after filtering
#     - Lists available columns in obs (cell metadata) and var (gene metadata)
#     - Prints available keys in uns (unstructured data)
#     """
    
#     # Load the AnnData object
#     adata = anndata.read_h5ad(file_path)

#     # Initial shape
#     initial_genes = adata.shape[1]
    
#     # Filter genes with at least one count
#     sc.pp.filter_genes(adata, min_counts=1)
#     filtered_genes = adata.shape[1]

#     print(f"\n📊 Dataset Shape: {adata.shape} (Cells × Genes)")
#     print(f"🧬 Genes Before Filtering: {initial_genes}")
#     print(f"🧬 Genes After Filtering: {filtered_genes}")

#     # Cell metadata summary
#     print("\n🔍 Available columns in obs (Cell Metadata):")
#     if not adata.obs.empty:
#         print(list(adata.obs.columns))
#     else:
#         print("⚠️ No columns found in obs.")

#     # Gene metadata summary
#     print("\n🔬 Available columns in var (Gene Metadata):")
#     if not adata.var.empty:
#         print(list(adata.var.columns))
#     else:
#         print("⚠️ No columns found in var.")

#     # Unique labels in obs (if available)
#     if "labels" in adata.obs.columns:
#         print("\n🏷️ Unique Labels in obs['labels']:")
#         print(adata.obs["labels"].value_counts())

#     #gene identifiers if any
#     if "gene_ids" in adata.var.index:
#         print("\n🧬 Gene Identifiers in var['gene_ids']:")
#         print(adata.var["gene_ids"].value_counts())

#     # Unstructured metadata
#     print("\n📦 Available keys in uns (Unstructured Data):")
#     if adata.uns:
#         print(list(adata.uns.keys()))
#     else:
#         print("⚠️ No keys found in uns.")

if __name__ == "__main__":
    # Example usage
    # print_adata_stats("extracted_csv/GSM2230757_human1_umifm_counts_human.h5ad")
    # print_adata_stats("extracted_csv/GSM2230758_human2_umifm_counts_human.h5ad")
    # print_adata_stats("../datasets/baron_2016h.h5ad")
    # print_adata_stats("/home/thechristyjo/Documents/Thesis/datasets/baron_2016h.h5ad")
    # print_adata_stats("/home/thechristyjo/Documents/Thesis/datasets/xin_2016.h5ad")
    analyze_adata(
        "/home/thechristyjo/Documents/Thesis/datasets/baron_2016h.h5ad",
        title="Baron 2016 Dataset Analysis"
    )
    analyze_adata(
        "/home/thechristyjo/Documents/Thesis/datasets/xin_2016.h5ad",
        title="Xin 2016 Dataset Analysis"
    )
    analyze_adata(
        "/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad",
        title="Concatenated scGPT Dataset Analysis"
    )