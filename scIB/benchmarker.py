import numpy as np
import scanpy as sc

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import matplotlib.pyplot as plt

import scvi.model as scvimodel
import os
from plottable import Table
import pandas as pd
import multiprocessing
num_workers = multiprocessing.cpu_count()

# %matplotlib inline


def benchmark_scib(adata_path):
    """
    Benchmark the given AnnData object using scIB metrics.
    """


    adata = sc.read(adata_path)
    adata.uns["filename"] = os.path.splitext(os.path.basename(adata_path))[0]

    # Check number of non-zero entries
    print("Non-zero entries in adata.X:", adata.X.nnz if hasattr(adata.X, 'nnz') else np.count_nonzero(adata.X))

    # Check basic stats- important for serring adata raw
    print("Max value:", adata.X.max())
    print("Min value:", adata.X.min())

    # Ensure counts are integers
    # adata.X = adata.X.astype(int)
    adata.X = adata.X.astype(np.float32)

    # Store a copy of the raw counts
    adata.raw = adata.copy()


    #logging
    print("Available embeddings:", adata.obsm.keys())
    print(adata.raw is not None)
    print("Available layers:", adata.layers.keys())
    if adata.raw is not None:
        print("adata.raw.X exists and has shape:", adata.raw.X.shape)
    else:
        print("adata.raw is None")

    # Check if 'counts' is in adata.layers
    if "counts" in adata.layers.keys():
        print("adata.layers['counts'] exists and has shape:", adata.layers["counts"].shape)
    else:
        print("'counts' not found in adata.layers")
    # print(adata.obsm["X_scGPT"].shape)
    # print(np.isnan(adata.obsm["X_scGPT"]).sum())
    print(adata.obsm["X_uce"].shape)
    print(np.isnan(adata.obsm["X_uce"]).sum())

    print("Avalable layers are: ", adata.layers.keys())


    print(type(adata.X))
    print(adata.X[:5, :5].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :5])


    print(adata.raw is not None)
    if adata.raw is not None:
        print(adata.raw.X[:5, :5])

    # adata.obs["cell_type"] = adata.obs["labels"] #use for the datasets like baron xin
    adata.obs["cell_type"] = adata.obs["cell_type"]
    print(adata.obs["cell_type"].value_counts())

    #computing the scvi metrics
    # scvimodel.SCVI.setup_anndata(adata, layer="counts", batch_key="batch_id")
    # scvimodel.SCVI.setup_anndata(adata, batch_key="batch_id")
    scvimodel.SCVI.setup_anndata(adata, batch_key="source")
    vae = scvimodel.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=30)
    vae.train()
    adata.obsm["scVI"] = vae.get_latent_representation()


    #computing the scANVI metrics
    lvae = scvimodel.SCANVI.from_scvi_model(
        vae,
        adata=adata,
        labels_key="cell_type",
        unlabeled_category="Unknown",
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)
    adata.obsm["scANVI"] = lvae.get_latent_representation()

    #saving the adata object
    adata.write_h5ad("test_scGPT_scanvi_scvi_benchmark.h5ad")


    # sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch_id")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="source")
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    # print(adata.obs["batch_id"].value_counts())
    print(adata.obs["source"].value_counts())
    # print(adata.obs["cell_type"].value_counts())
    print(adata.obs.columns)
    # adata.obs["cell_type"] = adata.obs["labels"] #use for the datasets like baron xin
    adata.obs["cell_type"] = adata.obs["cell_type"]
    print(adata.obs["cell_type"].value_counts())

    # embedding_keys = ["Unintegrated", "scVI", "scANVI", "X_scGPT"]
    embedding_keys = ["Unintegrated", "scVI", "scANVI", "X_uce"]
    bm = Benchmarker(
        adata,
        # batch_key="batch_id",
        batch_key="source",
        # label_key="labels",
        label_key="cell_type",
        bio_conservation_metrics=BioConservation(
            # Use a subset of metrics that are less memory-intensive
            silhouette_label=False,clisi_knn=True
            # Skip silhouette which is causing the memory error
        ),
        batch_correction_metrics=BatchCorrection(),
        # embedding_obsm_keys=["X_scGPT"],
        # embedding_obsm_keys=["X_uce"],
        embedding_obsm_keys=embedding_keys,
        n_jobs=6,
    )
    bm.benchmark()
    print("Benchmarking complete.")
    print(bm.get_results())


    fig,ax= plt.subplots(figsize=(12, 5))  # Adjust size as needed
    table = bm.plot_results_table()     # Pass ax explicitly

    
    # Create a string with the embedding keys for the filename
    embedding_str = "_".join([key.replace("X_", "") for key in embedding_keys])

    # tab = Table(pd.DataFrame(table), ax=ax)
    print("the table type is ",type(table))

    table_fig = table.ax.figure 

    # Save the table's figure (not your empty fig)
    #get the basename of the adata file
    base_filename = adata.uns.get("filename", "adata") 

    # Create the new filename with embedding keys
    output_filename = f"scib_benchmark_{base_filename}_{embedding_str}"

     # Save with the new filename
    table_fig.savefig(f"{output_filename}.pdf", bbox_inches="tight")
    table_fig.savefig(f"{output_filename}.svg", dpi=300, bbox_inches="tight")
    
    # table_fig.savefig("table_output.png", dpi=300, bbox_inches="tight")
    # table_fig.savefig("table_output.pdf", bbox_inches="tight")
    plt.figure(table_fig.number)


    # plt.show()


if __name__ == "__main__":
    # Load your AnnData object
    # adata = sc.read("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad")

    # Run the benchmark
    # benchmark_scib(adata)
    # benchmark_scib("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad")
    # benchmark_scib("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_chen_2017_hrvatin_2018.h5ad")
    # benchmark_scib("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_macosko_2015_shekhar_2016.h5ad")
    # benchmark_scib("/home/thechristyjo/Documents/Thesis/test_scGPT_scanvi_scvi_benchmark.h5ad")
    
    # when checking with the uce datasets
    # benchmark_scib("F:/Thesis/adata_concat_uce_baron_2016h_uce_adata_xin_2016_uce_adata.h5ad")
    benchmark_scib("F:/Thesis/adata_concat_UCE_sample_proc_lung_train_uce_adata_sample_proc_lung_test_uce_adata.h5ad")