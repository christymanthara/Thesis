# Quantifying Distribution Differences Between Single-Cell RNA-Seq Datasets Using Divergence Metrics

## Introduction

Single-cell RNA sequencing (scRNA-seq) studies often require comparison between datasets from different experimental conditions, technologies, or even species. To quantitatively assess the differences between dataset distributions in low-dimensional embeddings, we implemented and analyzed two information-theoretic metrics: **Kullback-Leibler (KL) divergence** and **Jensen-Shannon (JS) divergence**. These metrics provide a rigorous mathematical framework for comparing probability distributions that represent cell states in the dimensionally-reduced embedding space.

## Methods

We performed dimension reduction via **Principal Component Analysis (PCA)** on preprocessed and normalized scRNA-seq datasets. The comparison of dataset distributions was conducted using **Kernel Density Estimation (KDE)** in this reduced PCA space. We implemented both KL and JS divergence calculations with careful consideration of sampling and bandwidth parameters.

The divergence metrics were computed as follows:

1. **Kullback-Leibler Divergence**: Measures how one probability distribution ($P$) diverges from a second reference probability distribution ($Q$):

   $$ \text{KL}(P||Q) = \sum_x P(x) \log\left(\frac{P(x)}{Q(x)}\right) $$

2. **Jensen-Shannon Divergence**: A symmetrized and smoothed version of KL divergence:

   $$ \text{JS}(P||Q) = 0.5 \times \text{KL}(P||M) + 0.5 \times \text{KL}(Q||M) $$

   where \( M = 0.5 \times (P + Q) \)

For both metrics, **higher values indicate greater differences** between distributions, with JS divergence being bounded between 0 and \( \ln(2) \approx 0.693 \).

## Key Findings

### Bandwidth Parameter Optimization

We discovered that the bandwidth parameter in Kernel Density Estimation critically influences the calculated divergence values:

| **Bandwidth** | **Behavior and Interpretation** |
|---------------|----------------------------------|
| 0.01 (very small) | Produces extremely high KL values (194–200) and maximum JS values (0.693), suggesting overfitting to dataset-specific noise rather than capturing biological differences |
| 0.1 (small) | Still produces high divergence values (KL: 67–84, JS: 0.693), indicating minimal distribution overlap |
| 0.5 (medium) | Provides biologically interpretable divergence values (KL: 3.9–4.6, JS: 0.54–0.63), capturing meaningful differences while acknowledging some distribution overlap |
| 1.0 (large) | Produces zero divergence values for all comparisons, indicating excessive smoothing that obscures all dataset differences |

Our analysis identified a **bandwidth of 0.5** as optimal for scRNA-seq dataset comparisons, providing a balance between sensitivity to differences and robustness to noise.

### Cross-Dataset Comparisons

We evaluated three dataset comparison scenarios with **bandwidth 0.5**:

1. **Human vs. Mouse Comparison**:
   - KL Divergence: **4.62**
   - JS Divergence: **0.63**
   - High values reflect the expected substantial differences between species in cellular molecular profiles.

2. **Human Pancreas vs. Human Pancreas (different studies)**:
   - KL Divergence: **4.15**
   - JS Divergence: **0.62**
   - The relatively high divergence highlights the substantial impact of batch effects and methodological differences.

3. **Baron vs. Xin Pancreas Datasets**:
   - KL Divergence: **3.92**
   - JS Divergence: **0.54**
   - The lower JS divergence indicates more shared structure between these datasets.

### Differential Sensitivity of KL vs. JS Divergence

A notable observation was the different sensitivity patterns of the two metrics:

- **KL Divergence** showed relatively smaller differences between comparison types (range: 3.92–4.62), focusing primarily on the most divergent aspects of the distributions.
- **JS Divergence** demonstrated greater discrimination (range: 0.54–0.63), especially for the Baron vs. Xin comparison, suggesting higher sensitivity to distribution similarities.

This behavior can be attributed to KL's **asymmetric nature**, which can be dominated by regions where \( P \gg Q \), while JS provides a **more balanced measure** through its symmetrized approach.

## Discussion

The divergence analysis provided several important insights:

1. **Cross-Species Differences**: Human vs. mouse comparisons showed the highest divergence, confirming the method's ability to detect fundamental biological differences.

2. **Batch Effect Quantification**: Substantial divergence values between human pancreas datasets from different studies highlight the significant impact of technical and experimental variation in scRNA-seq data.

3. **Dataset Compatibility**: Lower JS divergence observed between Baron and Xin datasets suggests better compatibility for integration tasks.

4. **Parameter Sensitivity**: A bandwidth of 0.5 was found to be appropriate, balancing sensitivity and robustness.

These information-theoretic metrics provide valuable **quantitative benchmarks** for assessing dataset similarity before integration or cross-dataset analysis. They complement traditional metrics like **ARI** and **AMI** by directly quantifying distribution similarities in the embedding space rather than just clustering agreement.

## Conclusion

KL and JS divergence metrics, when properly parameterized, provide **meaningful quantitative assessments** of scRNA-seq dataset differences. These metrics capture both **biological variation** (e.g., cross-species differences) and **technical variation** (batch effects), offering valuable benchmarks for dataset comparison and integration tasks.

The **differential sensitivity** of KL and JS divergence provides complementary insights, with **JS divergence** appearing particularly useful for identifying datasets with greater potential for successful integration.

