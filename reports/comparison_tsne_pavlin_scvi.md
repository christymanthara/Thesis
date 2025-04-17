# 🔬 Divergence Analysis: scVI vs PCA on Pancreatic Tissue Datasets

## 🧬 Datasets
All datasets used in this analysis are derived from **pancreatic tissue**. Two dataset pairs were analyzed:

1. **Baron (2016)** vs **Xin (2016)**
2. **GSM2230757 (Human1)** vs **GSM2230758 (Human2)**

Each pair was processed and embedded using two methods:
- **scVI** (nonlinear probabilistic model)
- **PCA** (linear dimensionality reduction with `n_components=50`)

---

## 📊 Divergence Metrics
The **KL Divergence** and **JS Divergence** were computed on the embedded latent spaces to assess distributional similarity.

| Tissue   | Dataset Pair         | Embedding | KL Divergence | JS Divergence | Interpretation |
|----------|----------------------|-----------|---------------|----------------|----------------|
| Pancreas | Baron vs Xin         | scVI      | 8.9544        | 0.6899         | High divergence; latent representations are distinct |
| Pancreas | Baron vs Xin         | PCA       | 3.9180        | 0.5406         | Lower divergence; PCA shows better alignment for these datasets |
| Pancreas | Human1 vs Human2     | scVI      | 8.3103        | 0.6877         | Strong divergence remains despite tissue similarity |
| Pancreas | Human1 vs Human2     | PCA       | 4.1533        | 0.6177         | PCA also shows divergence but less than scVI         |

---

## 📌 Key Insights

- **scVI consistently results in higher divergence values**, potentially because it retains more complex, biologically relevant information, including donor- or condition-specific variation.
- **PCA shows lower divergence**, possibly due to its linear nature, smoothing over subtle but meaningful differences.
- Despite all samples being from **the same tissue (pancreas)**, inter-individual and inter-study variation is strong enough to yield measurable divergence in both embeddings.

---

## ✅ Recommendations

- Consider combining **scVI with label supervision (scANVI)** to reduce unwanted divergence while maintaining biological signals.
- Use **per-cell-type analysis** to determine which populations drive the divergence.
- Further harmonization (e.g., gene selection, normalization strategy) may help align datasets more tightly.

---

*This comparative study highlights how embedding choice significantly impacts perceived similarity between single-cell datasets, even from the same tissue.*

