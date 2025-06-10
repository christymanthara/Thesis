import anndata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def tune_knn_hyperparameters(
    adata=None,
    embedding_key=None,
    ref_source_name="baron_2016h",
    query_source_name="xin_2016",
    labels_key="labels",
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    max_k=30,
    cv_folds=5,
    plot=True,
    save_plots=False,
    output_dir='./'
):
    """
    Tune KNN hyperparameters by testing different K values and evaluating performance.

    This function can be used in two ways:
    1. By providing an AnnData object (`adata`) and specifying the `embedding_key`.
       The function will then split the data into training and test sets based on
       the `source` column in `adata.obs`.
    2. By providing pre-split training and test data (`X_train`, `y_train`, etc.).

    Parameters:
    -----------
    adata : anndata.AnnData, optional
        An AnnData object containing the embeddings and metadata.
        If provided, the function will extract data from here.
    embedding_key : str, optional
        The key in `adata.obsm` or `adata.layers` corresponding to the embedding to use.
        Required if `adata` is provided.
    ref_source_name : str, default="baron_2016h"
        The value in `adata.obs['source']` identifying the reference/training data.
    query_source_name : str, default="xin_2016"
        The value in `adata.obs['source']` identifying the query/test data.
    labels_key : str, default="labels"
        The key in `adata.obs` for the cell type labels.
    X_train : array-like, optional
        Training features. Required if `adata` is not provided.
    y_train : array-like, optional
        Training labels. Required if `adata` is not provided.
    X_test : array-like, optional
        Test features. Required if `adata` is not provided.
    y_test : array-like, optional
        Test labels. Required if `adata` is not provided.
    max_k : int, default=30
        Maximum number of neighbors to test (tests K from 1 to max_k).
    cv_folds : int, default=5
        Number of cross-validation folds for more robust error estimation.
    plot : bool, default=True
        Whether to display the accuracy and error rate plots.
    save_plots : bool, default=False
        Whether to save plots as PDF files.
    output_dir : str, default='./'
        Directory to save PDF files (if save_plots=True).

    Returns:
    --------
    dict : A dictionary containing comprehensive results, including best K values,
           accuracies, error rates, and paths to saved plots.
    """
    # --- Input Validation and Data Preparation ---
    if adata is not None:
        print("AnnData object provided. Extracting training and test sets.")
        if embedding_key is None:
            raise ValueError("`embedding_key` must be provided when using an AnnData object.")

        # Extract embedding data from .obsm or .layers
        if embedding_key in adata.obsm:
            embedding_data = adata.obsm[embedding_key]
        elif embedding_key in adata.layers:
            embedding_data = adata.layers[embedding_key]
        else:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm or adata.layers.")

        # Create masks for splitting data
        reference_mask = adata.obs["source"] == ref_source_name
        query_mask = adata.obs["source"] == query_source_name

        if not np.any(reference_mask) or not np.any(query_mask):
            raise ValueError("Could not find data for reference or query source names in adata.obs['source'].")

        # Assign data to train/test variables
        X_train = embedding_data[reference_mask]
        y_train = adata.obs[labels_key][reference_mask].values.astype(str)
        X_test = embedding_data[query_mask]
        y_test = adata.obs[labels_key][query_mask].values.astype(str)
        print(f"Data extracted using embedding: '{embedding_key}'")
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")

    elif all(arg is not None for arg in [X_train, y_train, X_test, y_test]):
        print("Using pre-split data for hyperparameter tuning.")
    else:
        raise ValueError("You must provide either an AnnData object and embedding_key OR all of X_train, y_train, X_test, and y_test.")

    # --- Core Hyperparameter Tuning Logic ---
    k_values = range(1, max_k + 1)
    mean_acc = np.zeros(max_k)
    std_acc = np.zeros(max_k)
    test_acc = np.zeros(max_k)
    error_rate = np.zeros(max_k)

    print(f"\nTesting K values from 1 to {max_k}...")

    for i, k in enumerate(k_values):
        neigh = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(neigh, X_train, y_train, cv=cv_folds, scoring='accuracy')
        mean_acc[i] = cv_scores.mean()
        std_acc[i] = cv_scores.std()

        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        test_acc[i] = accuracy_score(y_test, y_pred)
        error_rate[i] = np.mean(y_pred != y_test)

    # Find best K values
    best_cv_idx = np.argmax(mean_acc)
    best_k_cv = k_values[best_cv_idx]
    best_acc_cv = mean_acc[best_cv_idx]

    best_test_idx = np.argmax(test_acc)
    best_k_test = k_values[best_test_idx]
    best_acc_test = test_acc[best_test_idx]

    best_error_idx = np.argmin(error_rate)
    best_k_error = k_values[best_error_idx]
    min_error = error_rate[best_error_idx]

    # --- Plotting ---
    plot_files = []
    if plot or save_plots:
        os.makedirs(output_dir, exist_ok=True)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle("KNN Hyperparameter Tuning Analysis", fontsize=16)

        # Plot 1: Cross-validation results
        ax1.plot(k_values, mean_acc, 'b-', linewidth=2, label='CV Mean Accuracy')
        ax1.fill_between(k_values, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3, color='blue', label='±1 std')
        ax1.axvline(x=best_k_cv, color='red', linestyle='--', alpha=0.7, label=f'Best K (CV) = {best_k_cv}')
        ax1.set(xlabel='Number of Neighbors (K)', ylabel='Cross-Validation Accuracy', title='Cross-Validation Accuracy vs. K')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Test set accuracy results
        ax2.plot(k_values, test_acc, 'g-', linewidth=2, label='Test Set Accuracy')
        ax2.axvline(x=best_k_test, color='red', linestyle='--', alpha=0.7, label=f'Best K (Test) = {best_k_test}')
        ax2.set(xlabel='Number of Neighbors (K)', ylabel='Test Set Accuracy', title='Test Set Accuracy vs. K')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Error rate results
        ax3.plot(k_values, error_rate, color='purple', linestyle='dashed', marker='o', markerfacecolor='red', markersize=5, label='Error Rate')
        ax3.axvline(x=best_k_error, color='red', linestyle='--', alpha=0.7, label=f'Min Error K = {best_k_error}')
        ax3.set(xlabel='Number of Neighbors (K)', ylabel='Error Rate', title='Error Rate vs. K')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_plots:
            filename = os.path.join(output_dir, 'knn_hyperparameter_tuning_{embedding_key}.pdf')
            plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            plot_files.append(filename)
            print(f"\nHyperparameter tuning plot saved as: {filename}")

        if plot:
            plt.show()
        else:
            plt.close() # Free memory if not showing

    # --- Results Summary ---
    print(f"\n=== Cross-Validation Results ===")
    print(f"Best CV accuracy: {best_acc_cv:.4f} (±{std_acc[best_cv_idx]:.4f}) with K={best_k_cv}")
    print(f"\n=== Test Set Results ===")
    print(f"Best test accuracy: {best_acc_test:.4f} with K={best_k_test}")
    print(f"\n=== Error Rate Results ===")
    print(f"Minimum error rate: {min_error:.4f} with K={best_k_error}")

    if best_k_cv == best_k_test == best_k_error:
        print(f"\n🎯 All methods agree! Recommended K = {best_k_cv}")
    else:
        print(f"\n📊 Different methods suggest different K values:")
        print(f"   - Cross-validation (most reliable) recommends K = {best_k_cv}")
        print(f"   - Test accuracy recommends K = {best_k_test}")
        print(f"   - Error rate recommends K = {best_k_error}")

    return {
        'k_values': np.array(k_values),
        'mean_accuracies': mean_acc,
        'std_accuracies': std_acc,
        'test_accuracies': test_acc,
        'error_rates': error_rate,
        'best_k': best_k_cv,
        'best_accuracy': best_acc_cv,
        'best_test_k': best_k_test,
        'best_test_accuracy': best_acc_test,
        'best_error_k': best_k_error,
        'min_error_rate': min_error,
        'plot_files': plot_files
    }

if __name__ == '__main__':
    # --- Example Usage ---
    
    anndata_file = "baron_2016h_xin_2016_preprocessed_uce_adata_X_scvi_X_scanvi_X_uce_test.h5ad"
    # Load the AnnData object
    adata = anndata.read_h5ad(anndata_file)

    # Call the function with the arrays
    results_from_arrays = tune_knn_hyperparameters(
        adata=adata,
        embedding_key='X_uce',
        ref_source_name="baron_2016h",
        query_source_name="xin_2016",
        max_k=20,
        plot=True,
        save_plots=True
    )
    print("\n--- Completed Finding optimal K and plotting ---")