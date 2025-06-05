#!/usr/bin/env python3
"""
Single-cell embedding benchmarking script.

Usage: python benchmark_script.py <adata_file> [options]

This script performs benchmarking of single-cell embeddings by:
1. Loading single-cell data from h5ad file
2. Running multiple subsampled benchmarks
3. Computing mean and std of benchmark scores across samples
"""

import argparse
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import faiss
import warnings
from tqdm.auto import tqdm

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
from scib_metrics.nearest_neighbors import NeighborsResults

# Suppress warnings
warnings.filterwarnings("ignore")


def faiss_hnsw_nn(X: np.ndarray, k: int):
    """GPU HNSW nearest neighbor search using faiss.

    See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    for index param details.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    M = 32
    index = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(X)
    distances, indices = gpu_index.search(X, k)
    del index
    del gpu_index
    # distances are squared
    return NeighborsResults(indices=indices, distances=np.sqrt(distances))


def faiss_brute_force_nn(X: np.ndarray, k: int):
    """GPU brute force nearest neighbor search using faiss."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(X.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(X)
    distances, indices = gpu_index.search(X, k)
    del index
    del gpu_index
    # distances are squared
    return NeighborsResults(indices=indices, distances=np.sqrt(distances))


def benchmark_single_sample(adata, label_key="cell_type", batch_key="sample_id", 
                           obsm_keys=["X_uce", "X_scGPT", "X_geneformer"], n_jobs=48):
    """
    Run benchmarking on a single sample of data.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing single-cell data
    label_key : str
        Key in adata.obs for cell type labels
    batch_key : str
        Key in adata.obs for batch information
    obsm_keys : list
        List of embedding keys in adata.obsm to benchmark
    n_jobs : int
        Number of parallel jobs for benchmarking
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing benchmark scores
    """
    print(f"Running benchmarking using cell type key: {label_key}")
    print(f"Running benchmarking using batch key: {batch_key}")
    print(f"Embedding keys: {obsm_keys}")
    
    biocons = BioConservation()
    batchcons = BatchCorrection(pcr_comparison=False)
    
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=None,
        n_jobs=n_jobs,
    )
    bm.prepare(neighbor_computer=faiss_brute_force_nn)
    bm.benchmark()
    df = bm.get_results(min_max_scale=False)
    return df


def run_benchmarking(adata_file, label_key="supercluster", batch_key="donor_id", 
                    obsm_keys=["X_uce", "X_scGPT", "X_geneformer"],
                    sample_size=100000, n_samples=10, n_jobs=48,
                    output_file=None):
    """
    Run complete benchmarking pipeline on single-cell data.
    
    Parameters:
    -----------
    adata_file : str
        Path to h5ad file containing single-cell data
    label_key : str
        Key in adata.obs for cell type labels (default: "supercluster")
    batch_key : str
        Key in adata.obs for batch information (default: "donor_id")
    obsm_keys : list
        List of embedding keys in adata.obsm to benchmark
    sample_size : int
        Number of cells to sample for each benchmark run
    n_samples : int
        Number of subsampled benchmarks to run
    n_jobs : int
        Number of parallel jobs for benchmarking
    output_file : str, optional
        Path to save results CSV file
        
    Returns:
    --------
    tuple
        (mean_scores, std_scores) - DataFrames with mean and std benchmark scores
    """
    print(f"Loading data from: {adata_file}")
    
    # Load the data
    try:
        adata = sc.read(adata_file, cache=True)
    except Exception as e:
        print(f"Error loading file {adata_file}: {e}")
        return None, None
    
    print(f"Data loaded successfully. Shape: {adata.shape}")
    print(f"Number of unique cell types ({label_key}): {len(adata.obs[label_key].unique())}")
    print(f"Number of unique batches ({batch_key}): {len(adata.obs[batch_key].unique())}")
    
    # Check if embedding keys exist
    available_keys = list(adata.obsm.keys())
    missing_keys = [key for key in obsm_keys if key not in available_keys]
    if missing_keys:
        print(f"Warning: Missing embedding keys: {missing_keys}")
        print(f"Available keys: {available_keys}")
        obsm_keys = [key for key in obsm_keys if key in available_keys]
        if not obsm_keys:
            print("Error: No valid embedding keys found!")
            return None, None
    
    print(f"Running {n_samples} subsampled benchmarks with {sample_size} cells each...")
    
    sample_score_dfs = []
    
    for i in tqdm(range(n_samples), desc="Running benchmarks"):
        # Create subsample with random state i
        subsample_adata = sc.pp.subsample(adata, copy=True, n_obs=sample_size, random_state=i)
        
        # Run benchmark on this sample
        sample_df = benchmark_single_sample(
            subsample_adata, 
            label_key=label_key, 
            batch_key=batch_key,
            obsm_keys=obsm_keys,
            n_jobs=n_jobs
        )
        
        print(f"Sample {i+1}/{n_samples} completed. Shape: {subsample_adata.shape}")
        sample_score_dfs.append(sample_df)
    
    # Compute mean and std across all samples
    print("Computing statistics across samples...")
    
    # Concatenate all results and group by embedding
    all_results = pd.concat([df.drop("Metric Type", errors='ignore').reset_index() 
                           for df in sample_score_dfs])
    
    grouped_mean = all_results.groupby("Embedding").agg(np.mean)
    grouped_std = all_results.groupby("Embedding").agg(np.std)
    
    print("\nMean benchmark scores:")
    print(grouped_mean)
    
    if "Bio conservation" in grouped_mean.columns:
        print(f"\nBio conservation scores (mean):")
        print(grouped_mean["Bio conservation"])
    
    # Save results if output file specified
    if output_file:
        print(f"\nSaving results to {output_file}")
        with pd.ExcelWriter(output_file) as writer:
            grouped_mean.to_excel(writer, sheet_name='Mean_Scores')
            grouped_std.to_excel(writer, sheet_name='Std_Scores')
        print("Results saved successfully!")
    
    return grouped_mean, grouped_std


def main():
    """Main function to handle command line arguments and run benchmarking."""
    parser = argparse.ArgumentParser(
        description="Single-cell embedding benchmarking script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_script.py data.h5ad
  python benchmark_script.py data.h5ad --label-key cell_type --batch-key sample_id
  python benchmark_script.py data.h5ad --sample-size 50000 --n-samples 5
  python benchmark_script.py data.h5ad --output results.xlsx
        """
    )
    
    parser.add_argument("adata_file", help="Path to h5ad file containing single-cell data")
    parser.add_argument("--label-key", default="labels", 
                       help="Key in adata.obs for cell type labels (default: supercluster)")
    parser.add_argument("--batch-key", default="batch_id",
                       help="Key in adata.obs for batch information (default: donor_id)")
    parser.add_argument("--obsm-keys", nargs="+", 
                       default=["X_uce", "X_scGPT", "X_geneformer"],
                       help="List of embedding keys in adata.obsm to benchmark")
    parser.add_argument("--sample-size", type=int, default=100000,
                       help="Number of cells to sample for each benchmark run (default: 100000)")
    parser.add_argument("--n-samples", type=int, default=10,
                       help="Number of subsampled benchmarks to run (default: 10)")
    parser.add_argument("--n-jobs", type=int, default=48,
                       help="Number of parallel jobs for benchmarking (default: 48)")
    parser.add_argument("--output", "-o", 
                       help="Output file path to save results (Excel format)")
    
    args = parser.parse_args()
    
    # Run benchmarking
    mean_scores, std_scores = run_benchmarking(
        adata_file=args.adata_file,
        label_key=args.label_key,
        batch_key=args.batch_key,
        obsm_keys=args.obsm_keys,
        sample_size=args.sample_size,
        n_samples=args.n_samples,
        n_jobs=args.n_jobs,
        output_file=args.output
    )
    
    if mean_scores is not None:
        print("\nBenchmarking completed successfully!")
    else:
        print("\nBenchmarking failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()