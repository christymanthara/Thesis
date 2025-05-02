import numpy as np
import scanpy as sc

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import matplotlib.pyplot as plt

import scvi.model as scvimodel
import os
from plottable import Table
import pandas as pd

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
    adata.X = adata.X.astype(int)

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
    print(adata.obsm["X_scGPT"].shape)
    print(np.isnan(adata.obsm["X_scGPT"]).sum())
    print("Avalable layers are: ", adata.layers.keys())


    print(type(adata.X))
    print(adata.X[:5, :5].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :5])


    print(adata.raw is not None)
    if adata.raw is not None:
        print(adata.raw.X[:5, :5])

    adata.obs["cell_type"] = adata.obs["labels"]
    print(adata.obs["cell_type"].value_counts())

    #computing the scvi metrics
    # scvimodel.SCVI.setup_anndata(adata, layer="counts", batch_key="batch_id")
    scvimodel.SCVI.setup_anndata(adata, batch_key="batch_id")
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
    

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch_id")
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    print(adata.obs["batch_id"].value_counts())
    # print(adata.obs["cell_type"].value_counts())
    print(adata.obs.columns)
    adata.obs["cell_type"] = adata.obs["labels"]
    print(adata.obs["cell_type"].value_counts())


    bm = Benchmarker(
        adata,
        batch_key="batch_id",
        label_key="labels",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["Unintegrated","scVI","scANVI","X_scGPT"],
        n_jobs=2,
    )
    bm.benchmark()
    print("Benchmarking complete.")
    print(bm.get_results())


    fig,ax= plt.subplots(figsize=(12, 5))  # Adjust size as needed
    table = bm.plot_results_table()     # Pass ax explicitly

    

    # tab = Table(pd.DataFrame(table), ax=ax)
    print("the table type is ",type(table))

    table_fig = table.ax.figure 

    # Save the table's figure (not your empty fig)
    filename = adata.uns.get("filename", "adata") 
    table_fig.savefig(f"scib_benchmark_results_{filename}.pdf", bbox_inches="tight")
    table_fig.savefig(f"scib_benchmark_results_{filename}.svg", dpi=300,bbox_inches="tight")
    
    # table_fig.savefig("table_output.png", dpi=300, bbox_inches="tight")
    # table_fig.savefig("table_output.pdf", bbox_inches="tight")
    plt.figure(table_fig.number)


    # plt.show()


if __name__ == "__main__":
    # Load your AnnData object
    # adata = sc.read("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad")
    # adata = sc.read("/home/thechristyjo/Documents/Thesis/shekhar_2016_scGPT.h5ad")
    # adata = sc.read("/home/thechristyjo/Documents/Thesis/datasets/baron_2016h.h5ad")
    # Run the benchmark
    # benchmark_scib(adata)
    # benchmark_scib("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad")
    # benchmark_scib("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_chen_2017_hrvatin_2018.h5ad")
    benchmark_scib("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_macosko_2015_shekhar_2016.h5ad")
    # benchmark_scib("/home/thechristyjo/Documents/Thesis/test_scGPT_scanvi_scvi_benchmark.h5ad")