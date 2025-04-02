```markdown
# Thesis Repository

This repository serves as the main repository for my thesis, which focuses on comparing batch-effect correction methods in single-cell genomics. It contains submodules for various aspects of the research, including referenced papers, implementations, datasets, and benchmark papers.

## Repository Structure

The repository is organized into the following submodules:

- **`papers/referenced`**: Contains the papers that are referenced in the thesis.
- **`scripts`**: Includes the implementations of the methods discussed in the referenced papers.
- **`datasets`**: Contains the datasets used for analysis in the thesis.
- **`benchmark/papers`**: Includes the papers that will be analyzed for benchmarking purposes.

## Cloning the Repository

To clone this repository along with its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/christymanthara/Thesis.git
```

To update all submodules to the latest commit, use the following command:

```bash
git submodule update --remote
```

For datasets, use the link below:
[Hemberg Lab scRNA-seq Datasets - Human Pancreas](https://hemberg-lab.github.io/scRNA.seq.datasets/human/pancreas/)

## Overview

This repository contains various scripts for processing, analyzing, and visualizing single-cell RNA sequencing (scRNA-seq) data using AnnData, Scanpy, openTSNE, and UMAP.

Make sure that you download the datasets and follow the instructions in the datasets subrepo before you proceed with the scripts here.

## Scripts

### `csv_h5ad.py`
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

### `pavlin_preprocess_plot_on_top.py`
Preprocesses gene expression data and computes embeddings.
- Normalizes and log-transforms the data.
- Selects genes based on variance.
- Computes PCA and t-SNE embeddings.
- Saves the processed data as .h5ad files.


### `umap_plot.py`
Computes and plots UMAP embeddings for two datasets.
- Loads and preprocesses two .h5ad datasets.
- Computes UMAP projections.
- Saves the UMAP plot as a PDF.

### `show_batch_effect_1_concatenated_datasets.py`
Visualizes batch effects using t-SNE or UMAP.
- Calls `pavlin_alignment.py` for t-SNE visualization.
- Calls `umap_plot.py` for UMAP visualization.
- Allows switching between t-SNE and UMAP methods.

## Batch Effects2

## Usage

Run each script individually with Python. Modify file paths as needed for specific datasets.

```bash
python scripts/scv_h5ad.py
python scripts/data_stats.py
python scripts/pavlin_preprocess_plot_on_top.py
python scripts/show_batch_effect_1_concatenated_datasets.py
```

### Data Utils
This is a package that handles the common preprocessing steps used in the analysis.

### Note
Change normalization from million to 10,000 when processing datasets.


