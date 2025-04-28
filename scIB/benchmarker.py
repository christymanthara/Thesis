import numpy as np
import scanpy as sc

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# %matplotlib inline

def benchmark_scib(adata):
    """
    Benchmark the given AnnData object using scIB metrics.
    """
    bm = Benchmarker(
        adata,
        batch_key="batch_id",
        label_key="labels",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["X_scGPT"],
        n_jobs=6,
    )
    bm.benchmark()

    bm.plot_results_table()


if __name__ == "__main__":
    # Load your AnnData object
    adata = sc.read("/home/thechristyjo/Documents/Thesis/adata_concat_scGPT_baron_2016h_xin_2016.h5ad")

    # Run the benchmark
    benchmark_scib(adata)