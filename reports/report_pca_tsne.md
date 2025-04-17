# Embedding Comparison Analysis Report

## Overview
This report analyzes the Kullback-Leibler (KL) and Jensen-Shannon (JS) divergence metrics obtained after PCA dimensionality reduction to 50 components across two dataset pairs:
- Baron vs. Xin datasets
- Human 1 vs. Human 2 datasets

## 📊 Visualizations

### 🔹 t-SNE Plot: Baron vs Xin

![t-SNE Baron vs Xin](./figures/Pavlins_tsne_with_metrics_genefiltered_baron_2016h_xin_2016.svg)

### 🔹 t-SNE Plot: Human1 vs Human2

![t-SNE Human vs Human](./figures/Pavlins_tsne_with_metrics_genefiltered_GSM2230757_human1_umifm_counts_human_GSM2230758_human2_umifm_counts_human.svg)

## Results Summary

| Dataset Comparison | KL Divergence | JS Divergence |
|--------------------|---------------|---------------|
| Baron vs. Xin      | 3.9180        | 0.5406        |
| Human 1 vs. Human 2| 4.1533        | 0.6177        |

## Analysis

### Divergence Metrics Interpretation
- **KL Divergence**: Measures how one probability distribution diverges from another. Higher values indicate greater difference between distributions.
- **JS Divergence**: A symmetrized and smoothed version of KL divergence, bounded between 0 and 1. Higher values indicate greater dissimilarity.

### Key Observations
1. The Human 1 vs. Human 2 comparison shows higher divergence values (KL: 4.1533, JS: 0.6177) than the Baron vs. Xin comparison (KL: 3.9180, JS: 0.5406).
2. Both comparisons show relatively high JS divergence values (>0.5), suggesting substantial differences between the compared datasets.
3. The PCA reduction to 50 components has maintained significant distributional differences between datasets.

### Potential Implications
- The higher divergence in the human datasets might indicate greater biological or technical variation between these samples.
- The Baron and Xin datasets, while still different, show slightly more similarity in their reduced-dimension representations.
- These divergence metrics suggest that even after dimensionality reduction, the distinct characteristics of each dataset remain identifiable.

## Conclusion
The applied PCA transformation to 50 dimensions preserves meaningful differences between datasets as measured by KL and JS divergence metrics. The human datasets show greater divergence than the Baron-Xin comparison, which could reflect underlying biological differences, technical variation, or batch effects. Further investigation into the specific features driving these differences could provide additional insights.

## Recommendations
1. Consider exploring which principal components contribute most to the observed divergence.
2. Evaluate whether batch correction methods might reduce the divergence if technical variation is suspected.
3. Assess if increasing or decreasing the number of PCA components impacts the divergence metrics.