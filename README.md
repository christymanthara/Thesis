# Thesis Repository

This repository serves as the main repository for my thesis, which focuses on comparing batch-effect correction methods in single-cell genomics. It contains submodules for various aspects of the research, including referenced papers, implementations, datasets, and benchmark papers.

## Repository Structure

The repository is organized into the following submodules:

- **`papers/referenced`**: Contains the papers that are referenced in the thesis.
- **`implementation`**: Includes the implementations of the methods discussed in the referenced papers.
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
for datasets use the link below
https://hemberg-lab.github.io/scRNA.seq.datasets/human/pancreas/


change normalization from million to a 10000