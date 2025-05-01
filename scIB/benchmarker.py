import numpy as np
import scanpy as sc

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import matplotlib.pyplot as plt



# %matplotlib inline

def benchmark_scib(adata):
    """
    Benchmark the given AnnData object using scIB metrics.
    """

    print("Available embeddings:", adata.obsm.keys())
    print(adata.obsm["X_scGPT"].shape)
    print(np.isnan(adata.obsm["X_scGPT"]).sum())

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch_id")
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    print(adata.obs["batch_id"].value_counts())
    # print(adata.obs["cell_type"].value_counts())
    print(adata.obs.columns)
    adata.obs["cell_type"] = adata.obs["labels"]
    print(adata.obs["cell_type"].value_counts())


    # bm = Benchmarker(
    #     adata,
    #     batch_key="batch_id",
    #     label_key="labels",
    #     bio_conservation_metrics=BioConservation(),
    #     batch_correction_metrics=BatchCorrection(),
    #     embedding_obsm_keys=["X_scGPT"],
    #     n_jobs=2,
    # )
    # bm.benchmark()
    # print("Benchmarking complete.")
    # print(bm.get_results())


    # bm.plot_results_table()
    # plt.show()


if __name__ == "__main__":
    # Load your AnnData object
    adata = sc.read("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad")

    # Run the benchmark
    benchmark_scib(adata)