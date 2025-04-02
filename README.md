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


# Batch effects 2
# Single-Cell RNA-seq Visualization Pipeline

This repository contains a set of Python scripts for preprocessing, transforming, and visualizing single-cell RNA sequencing (scRNA-seq) data using PCA and t-SNE dimensionality reduction techniques.

## Overview

The pipeline consists of three main scripts that should be executed in the following order:

1. `pavlin_preprocess_plot_on_top.py` - Preprocess scRNA-seq data and compute initial embeddings
2. `multiplot_transform_and_plot.py` - Transform new datasets into the same embedding space
3. `multiplot_tsne_pca_plot_on_top.py` - Generate comparative visualizations of multiple datasets

**Key Concept:**
The pipeline is designed to map secondary datasets into a reference embedding space by treating each data point independently. This approach intentionally disregards interactions present within the secondary dataset and prevents the formation of clusters that would be specific to the secondary data. By doing so, it enables direct comparison of cell populations across different studies within a unified reference space.

## Script Details

### 1. pavlin_preprocess_plot_on_top.py

This script processes raw scRNA-seq data and generates t-SNE embeddings at different gene set sizes:

**Functions:**
- `preprocess_anndata()`: Loads, normalizes, and scales scRNA-seq data
- `compute_pca()`: Performs PCA dimensionality reduction
- `compute_tsne()`: Computes t-SNE embeddings using multiscale approach
- `run_gene_preprocess_pipeline()`: Main pipeline that processes data at different gene counts (250, 3000, and all genes)

**Outputs:**
- `*_embedding_tsne_250_genes.h5ad`: Processed dataset with 250 highly variable genes
- `*_embedding_tsne_3000_genes.h5ad`: Processed dataset with 3000 highly variable genes
- `*_embedding_tsne_all_genes.h5ad`: Processed dataset with all genes

### 2. multiplot_transform_and_plot.py

This script transforms new scRNA-seq datasets into an existing embedding space, allowing for comparison across different studies:

**Functions:**
- `multiplot_transform()`: Projects new data onto existing t-SNE embeddings using different gene subset sizes

**Process:**
1. Loads reference and new datasets
2. Finds shared genes between datasets
3. Creates embeddings for different gene subset sizes (250, 1000, all)
4. Visualizes the embeddings by overlaying new data on reference data
5. **Importantly:** Each cell from the secondary dataset is mapped independently to the reference space, deliberately ignoring relationships between cells in the secondary dataset

**Outputs:**
- `new_embedding_tsne_250_genes.h5ad`: New dataset embedded using 250 genes
- `new_embedding_tsne_1000_genes.h5ad`: New dataset embedded using 1000 genes
- `new_embedding_tsne_all_genes.h5ad`: New dataset embedded using all shared genes
- A PDF visualization showing the embedding results

### 3. multiplot_tsne_pca_plot_on_top.py

This script creates publication-ready visualizations comparing PCA and t-SNE results across multiple datasets:

**Functions:**
- `get_common_prefix()`: Extracts common prefix from filenames for output naming
- `multiplot_tsne_pca()`: Generates PCA and t-SNE plots for each provided dataset

**Features:**
- Creates multi-panel figures with PCA and t-SNE visualizations
- Automatically adapts layout based on number of input files
- Adds alphabetical labels to panels
- Outputs a PDF with separate pages for PCA and t-SNE visualizations

**Outputs:**
- A PDF file containing visualizations of the PCA and t-SNE embeddings

## Usage Example

```bash
# Step 1: Preprocess the reference dataset and compute embeddings
python scripts/pavlin_preprocess_plot_on_top.py

# Step 2: Transform new dataset into the reference embedding space
python scripts/multiplot_transform_and_plot.py

# Step 3: Generate visualizations of the embeddings
python scripts/multiplot_tsne_pca_plot_on_top.py
```

## Data Organization

The scripts assume the following file structure:
```
project/
├── scripts/
│   ├── pavlin_preprocess_plot_on_top.py
│   ├── multiplot_transform_and_plot.py
│   ├── multiplot_tsne_pca_plot_on_top.py
│   └── utils.py
└── Datasets/
    ├── baron_2016h.h5ad
    └── xin_2016.h5ad
```

## Notes

- The scripts use a custom `utils.py` module that needs to be available in the same directory.
- The default datasets used are from Baron et al. 2016 and Xin et al. 2016 (pancreatic islet cell datasets).
- For large datasets, the computation might be memory-intensive and time-consuming.
- The independent mapping approach helps identify corresponding cell types across studies without being influenced by batch effects or dataset-specific cluster formations.

### Note
Change normalization from million to 10,000 when processing datasets.


