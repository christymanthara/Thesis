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
import os
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import faiss
import warnings
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

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


def clean_numeric_data(df):
    """
    Clean dataframe to ensure all values are numeric and handle NaN/inf values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe with numeric values only
    """
    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0 (or you could use df.dropna() to remove rows with NaN)
    df = df.fillna(0)
    
    return df


def benchmark_single_sample(adata, label_key="labels", batch_key="batch_id", 
                           obsm_keys=["X_uce", "X_scANVI", "X_scVI"], n_jobs=48):
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


def save_results_pdf(grouped_mean, grouped_std, output_file, adata_file, 
                    label_key, batch_key, obsm_keys, sample_size, n_samples):
    """
    Save benchmark results to PDF with visualizations and summary tables.
    
    Parameters:
    -----------
    grouped_mean : pandas.DataFrame
        Mean benchmark scores
    grouped_std : pandas.DataFrame
        Standard deviation of benchmark scores
    output_file : str
        Path to output PDF file
    adata_file : str
        Original data file path
    label_key : str
        Cell type label key used
    batch_key : str
        Batch key used
    obsm_keys : list
        Embedding keys benchmarked
    sample_size : int
        Sample size used for benchmarks
    n_samples : int
        Number of samples used
    """
    with PdfPages(output_file) as pdf:
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Page 1: Summary Information
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        summary_text = f"""
Single-cell Embedding Benchmarking Results

Dataset Information:
• Data file: {os.path.basename(adata_file)}
• Cell type key: {label_key}
• Batch key: {batch_key}
• Embeddings benchmarked: {', '.join(obsm_keys)}

Benchmarking Parameters:
• Sample size per benchmark: {sample_size:,} cells
• Number of benchmark samples: {n_samples}
• Total cells analyzed: {sample_size * n_samples:,}

Summary Statistics:
• Number of embeddings: {len(grouped_mean)}
• Metrics computed: {len(grouped_mean.columns)}
"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        plt.title('Benchmarking Summary', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Mean Scores Table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with mean scores
        table_data = grouped_mean.round(4)
        table = ax.table(cellText=table_data.values,
                        rowLabels=table_data.index,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(table_data)):
            table[(i+1, -1)].set_facecolor('#E8F5E8')
        
        plt.title('Mean Benchmark Scores', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Standard Deviation Table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with std scores
        table_data_std = grouped_std.round(4)
        table = ax.table(cellText=table_data_std.values,
                        rowLabels=table_data_std.index,
                        colLabels=table_data_std.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data_std.columns)):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(table_data_std)):
            table[(i+1, -1)].set_facecolor('#FFF3E0')
        
        plt.title('Standard Deviation of Benchmark Scores', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Bar plot of mean scores for each metric
        metrics = grouped_mean.columns
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot bars with error bars
            embeddings = grouped_mean.index
            means = grouped_mean[metric]
            stds = grouped_std[metric]
            
            bars = ax.bar(range(len(embeddings)), means, yerr=stds, 
                         capsize=5, alpha=0.7)
            ax.set_xlabel('Embeddings')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(range(len(embeddings)))
            ax.set_xticklabels(embeddings, rotation=45, ha='right')
            
            # Add value labels on bars
            for j, (bar, mean_val) in enumerate(zip(bars, means)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds.iloc[j]/2,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5: Heatmap of mean scores (with error handling)
        if len(grouped_mean) > 1 and len(grouped_mean.columns) > 1:
            try:
                # Clean the data before plotting
                clean_mean_data = clean_numeric_data(grouped_mean.copy())
                
                # Check if we have valid numeric data after cleaning
                if clean_mean_data.empty or clean_mean_data.isna().all().all():
                    print("Warning: No valid numeric data for heatmap, skipping heatmap page")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create heatmap with cleaned data
                    sns.heatmap(clean_mean_data, annot=True, fmt='.3f', 
                               cmap='RdYlBu_r', center=clean_mean_data.values.mean(), 
                               ax=ax, cbar_kws={'shrink': 0.8})
                    
                    ax.set_title('Heatmap of Mean Benchmark Scores', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Metrics')
                    ax.set_ylabel('Embeddings')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                print(f"Warning: Could not create heatmap due to error: {e}")
                print("Skipping heatmap page and continuing with other visualizations")


def run_benchmarking(adata_file, label_key="labels", batch_key="batch_id", 
                           obsm_keys=["X_uce", "X_scANVI", "X_scVI"],
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
        Path to save results (Excel or PDF based on extension)
        
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
    
    # Check if sample size is larger than available data
    total_cells = adata.n_obs
    if sample_size > total_cells:
        print(f"Warning: Requested sample size ({sample_size:,}) is larger than available cells ({total_cells:,})")
        print(f"Adjusting sample size to {total_cells:,} cells (using all available data)")
        sample_size = total_cells
    
    print(f"Running {n_samples} subsampled benchmarks with {sample_size:,} cells each...")
    
    sample_score_dfs = []
    
    for i in tqdm(range(n_samples), desc="Running benchmarks"):
        # Create subsample with random state i
        if sample_size < total_cells:
            # Only subsample if we have more cells than requested
            subsample_adata = sc.pp.subsample(adata, copy=True, n_obs=sample_size, random_state=i)
        else:
            # Use all data if sample_size >= total_cells
            subsample_adata = adata.copy()
        
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
    
    # Clean the data before computing statistics
    numeric_columns = all_results.select_dtypes(include=[np.number]).columns
    all_results_clean = all_results.copy()
    for col in numeric_columns:
        all_results_clean[col] = pd.to_numeric(all_results_clean[col], errors='coerce')
    
    grouped_mean = all_results_clean.groupby("Embedding").agg(np.mean, numeric_only=True)
    grouped_std = all_results_clean.groupby("Embedding").agg(np.std, numeric_only=True)
    
    # Clean the final results
    grouped_mean = clean_numeric_data(grouped_mean)
    grouped_std = clean_numeric_data(grouped_std)
    
    print("\nMean benchmark scores:")
    print(grouped_mean)
    
    if "Bio conservation" in grouped_mean.columns:
        print(f"\nBio conservation scores (mean):")
        print(grouped_mean["Bio conservation"])
    
    # Save results if output file specified, or generate default timestamped outputs
    if output_file:
        print(f"\nSaving results to {output_file}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Determine output format based on file extension
        file_ext = os.path.splitext(output_file)[1].lower()
        
        if file_ext == '.pdf':
            save_results_pdf(grouped_mean, grouped_std, output_file, adata_file,
                            label_key, batch_key, obsm_keys, sample_size, n_samples)
            print("PDF results saved successfully!")
            
        elif file_ext in ['.xlsx', '.xls']:
            with pd.ExcelWriter(output_file) as writer:
                grouped_mean.to_excel(writer, sheet_name='Mean_Scores')
                grouped_std.to_excel(writer, sheet_name='Std_Scores')
            print("Excel results saved successfully!")
            
        else:
            # Default to CSV if extension not recognized
            base_name = os.path.splitext(output_file)[0]
            # Ensure directory exists for CSV files too
            csv_dir = os.path.dirname(base_name)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            grouped_mean.to_csv(f"{base_name}_mean_scores.csv")
            grouped_std.to_csv(f"{base_name}_std_scores.csv")
            print("CSV results saved successfully!")
    
    else:
        # Generate default timestamped outputs when no --output flag is provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"benchmark_results_{timestamp}"
        
        print(f"\nNo output file specified. Generating default timestamped outputs...")
        print(f"Base filename: {base_filename}")
        
        # Save CSV files
        csv_mean_file = f"{base_filename}_mean_scores.csv"
        csv_std_file = f"{base_filename}_std_scores.csv"
        grouped_mean.to_csv(csv_mean_file)
        grouped_std.to_csv(csv_std_file)
        print(f"CSV files saved: {csv_mean_file}, {csv_std_file}")
        
        # Save PDF file
        pdf_file = f"{base_filename}.pdf"
        save_results_pdf(grouped_mean, grouped_std, pdf_file, adata_file,
                        label_key, batch_key, obsm_keys, sample_size, n_samples)
        print(f"PDF file saved: {pdf_file}")
        
        print("Default timestamped results saved successfully!")
    
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
  python benchmark_script.py data.h5ad --output results.pdf
        """
    )
    
    parser.add_argument("adata_file", help="Path to h5ad file containing single-cell data")
    parser.add_argument("--label-key", default="labels", 
                       help="Key in adata.obs for cell type labels (default: labels)")
    parser.add_argument("--batch-key", default="batch_id",
                       help="Key in adata.obs for batch information (default: batch_id)")
    parser.add_argument("--obsm-keys", nargs="+", 
                       default=["X_uce", "X_scANVI", "X_scVI"],
                       help="List of embedding keys in adata.obsm to benchmark")
    parser.add_argument("--sample-size", type=int, default=100000,
                       help="Number of cells to sample for each benchmark run (default: 100000)")
    parser.add_argument("--n-samples", type=int, default=10,
                       help="Number of subsampled benchmarks to run (default: 10)")
    parser.add_argument("--n-jobs", type=int, default=48,
                       help="Number of parallel jobs for benchmarking (default: 48)")
    parser.add_argument("--output", "-o", 
                       help="Output file path to save results (.xlsx/.xls for Excel, .pdf for PDF, other extensions default to CSV)")
    
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
    
    # cross validated
    
    
    
    # find 10 closest embedded neighbouyrs of B in A 