import numpy as np
import scanpy as sc

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# %matplotlib inline

def benchmark_scib(adata):
    """
    Benchmark the given AnnData object using scIB metrics.
    """
    
    
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True) #n_comps = 50 as used by pavlin, check with 30 and 50. we are not looking for any higly variable
    
    adata = adata[:, adata.var.highly_variable].copy()
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    
    bm = Benchmarker(
        adata,
        batch_key="batch_id",
        label_key="labels",
        bio_conservation_metrics=BioConservation(
            # Use a subset of metrics that are less memory-intensive
            ["nmi_cluster_labels", "graph_connectivity"],
            # Skip silhouette which is causing the memory error
            
        ),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["Unintegrated,X_uce"],
        n_jobs=1,  # Reduce parallelism to save memory
    )
    bm.benchmark()

    bm.plot_results_table()


if __name__ == "__main__":
    # Load your AnnData object
    # adata = sc.read("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad")
    adata = sc.read("F:/Thesis/UCE/baron_2016h_uce_adata.h5ad")

    # Run the benchmark
    benchmark_scib(adata)