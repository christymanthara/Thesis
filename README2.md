# Repository:  Thesis on studying batch effects

## Overview
This repository contains various scripts for processing, analyzing, and visualizing single-cell RNA sequencing (scRNA-seq) data using AnnData, Scanpy, openTSNE, and UMAP.

Make sure that you download the datasets and follow the instructions in the datasets subrepo before you proceed with the scripts here.

## Scripts

### `scv_h5ad.py`
Converts .csv.gz files extracted from a .tar archive into AnnData (.h5ad) format.
- Extracts .csv.gz files from a .tar archive.
- Converts expression data to a sparse matrix format.
- Assigns species labels (human/mouse) based on filenames.
- Saves the processed data as .h5ad files.

### `data_stats.py`
Prints key statistics about an AnnData (.h5ad) dataset.
- Displays dataset shape (Cells × Genes).
- Reports the number of genes before and after filtering.
- Summarizes cell and gene metadata.
- Lists unique labels in the observation metadata (if available).

### `pavlin_alignment.py`
Performs t-SNE visualization for comparing two datasets.
- Loads and preprocesses two .h5ad datasets.
- Computes t-SNE embeddings using openTSNE with multiscale perplexities.
- Saves the t-SNE plot as a PDF.

### `pavlin_plot_on_top.py`
Preprocesses gene expression data and computes embeddings.
- Normalizes and log-transforms the data.
- Selects genes based on variance.
- Computes PCA and t-SNE embeddings.
- Saves the processed data as .h5ad files.

### `show_batch_effect.py`
Visualizes batch effects using t-SNE or UMAP.
- Calls `pavlin_alignment.py` for t-SNE visualization.
- Calls `umap_plot.py` for UMAP visualization.
- Allows switching between t-SNE and UMAP methods.

### `umap_plot.py`
Computes and plots UMAP embeddings for two datasets.
- Loads and preprocesses two .h5ad datasets.
- Computes UMAP projections.
- Saves the UMAP plot as a PDF.

## Usage
Run each script individually with Python. Modify file paths as needed for specific datasets.

```bash
python scripts/scv_h5ad.py
python scripts/data_stats.py
python scripts/pavlin_alignment.py
python scripts/pavlin_plot_on_top.py
python scripts/show_batch_effect.py
python scripts/umap_plot.py
```
### Data Utils
This is a package that handles the common preprocessing steps we have used here.
