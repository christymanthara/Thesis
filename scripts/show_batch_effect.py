from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt

def show_batch_effects(data, labels, visual='tsne'):
    if visual == 'tsne':
        reducer = TSNE(n_components=2)
    elif visual == 'umap':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Invalid value for visual. Choose 'tsne' or 'umap'.")

    reduced_data = reducer.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Batch Effects Visualization using {visual.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()