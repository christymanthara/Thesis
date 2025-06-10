import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import os
from datetime import datetime

def tune_knn_hyperparameters(X_train, y_train, X_test, y_test, max_k=30, cv_folds=5, plot=True, save_plots=False, output_dir='./'):
    """
    Tune KNN hyperparameters by testing different K values and evaluating performance.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    max_k : int, default=30
        Maximum number of neighbors to test (tests K from 1 to max_k)
    cv_folds : int, default=5
        Number of cross-validation folds for more robust error estimation
    plot : bool, default=True
        Whether to display the accuracy and error rate plots
    save_plots : bool, default=False
        Whether to save plots as PDF files
    output_dir : str, default='./'
        Directory to save PDF files (if save_plots=True)
    
    Returns:
    --------
    dict : Dictionary containing results
        - 'k_values': Array of K values tested
        - 'mean_accuracies': Mean accuracy for each K
        - 'std_accuracies': Standard deviation of accuracy for each K
        - 'test_accuracies': Single test set accuracy for each K
        - 'error_rates': Error rate for each K on test set
        - 'best_k': Optimal K value based on cross-validation
        - 'best_accuracy': Best cross-validation accuracy
        - 'best_test_k': Optimal K value based on test set accuracy
        - 'best_test_accuracy': Best test set accuracy
        - 'best_error_k': Optimal K value based on minimum error rate
        - 'min_error_rate': Minimum error rate achieved
        - 'plot_files': List of saved plot file paths (if save_plots=True)
    """
    
    k_values = range(1, max_k + 1)
    mean_acc = np.zeros(max_k)
    std_acc = np.zeros(max_k)
    test_acc = np.zeros(max_k)
    error_rate = np.zeros(max_k)
    
    print(f"Testing K values from 1 to {max_k}...")
    
    for i, k in enumerate(k_values):
        # Create KNN classifier
        neigh = KNeighborsClassifier(n_neighbors=k)
        
        # Cross-validation for more robust accuracy estimation
        cv_scores = cross_val_score(neigh, X_train, y_train, cv=cv_folds, scoring='accuracy')
        mean_acc[i] = cv_scores.mean()
        std_acc[i] = cv_scores.std()
        
        # Calculate test set accuracy and error rate
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        test_acc[i] = metrics.accuracy_score(y_test, y_pred)
        error_rate[i] = np.mean(y_pred != y_test)  # Calculate error rate
    
    # Find best K values
    best_cv_idx = np.argmax(mean_acc)
    best_k_cv = k_values[best_cv_idx]
    best_acc_cv = mean_acc[best_cv_idx]
    
    best_test_idx = np.argmax(test_acc)
    best_k_test = k_values[best_test_idx]
    best_acc_test = test_acc[best_test_idx]
    
    # Find best K for minimum error rate
    best_error_idx = np.argmin(error_rate)
    best_k_error = k_values[best_error_idx]
    min_error = error_rate[best_error_idx]
    
    # Plot results
    plot_files = []
    if plot:
        # Create output directory if it doesn't exist
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combined plot with all three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Cross-validation results
        ax1.plot(k_values, mean_acc, 'b-', linewidth=2, label='CV Mean Accuracy')
        ax1.fill_between(k_values, 
                        mean_acc - std_acc, 
                        mean_acc + std_acc, 
                        alpha=0.3, color='blue', label='±1 std')
        ax1.fill_between(k_values, 
                        mean_acc - 2*std_acc, 
                        mean_acc + 2*std_acc, 
                        alpha=0.1, color='blue', label='±2 std')
        
        ax1.axvline(x=best_k_cv, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best K={best_k_cv}')
        ax1.set_xlabel('Number of Neighbors (K)')
        ax1.set_ylabel('Cross-Validation Accuracy')
        ax1.set_title('KNN Hyperparameter Tuning - Cross Validation Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test set accuracy results
        ax2.plot(k_values, test_acc, 'g-', linewidth=2, label='Test Set Accuracy')
        ax2.axvline(x=best_k_test, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best K={best_k_test}')
        ax2.set_xlabel('Number of Neighbors (K)')
        ax2.set_ylabel('Test Set Accuracy')
        ax2.set_title('KNN Hyperparameter Tuning - Test Set Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Error rate results
        ax3.plot(k_values, error_rate, color='blue', linestyle='dashed', 
                marker='o', markerfacecolor='red', markersize=6, linewidth=2, 
                label='Error Rate')
        ax3.axvline(x=best_k_error, color='red', linestyle='--', alpha=0.7, 
                   label=f'Min Error K={best_k_error}')
        ax3.set_xlabel('Number of Neighbors (K)')
        ax3.set_ylabel('Error Rate')
        ax3.set_title('KNN Hyperparameter Tuning - Error Rate vs K Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save combined plot
        if save_plots:
            combined_filename = os.path.join(output_dir, f'knn_tuning_combined_{timestamp}.pdf')
            plt.savefig(combined_filename, format='pdf', dpi=300, bbox_inches='tight')
            plot_files.append(combined_filename)
            print(f"Combined plot saved as: {combined_filename}")
        
        plt.show()
        
        # Individual plots
        if save_plots:
            # Cross-validation plot
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, mean_acc, 'b-', linewidth=2, label='CV Mean Accuracy')
            plt.fill_between(k_values, 
                            mean_acc - std_acc, 
                            mean_acc + std_acc, 
                            alpha=0.3, color='blue', label='±1 std')
            plt.fill_between(k_values, 
                            mean_acc - 2*std_acc, 
                            mean_acc + 2*std_acc, 
                            alpha=0.1, color='blue', label='±2 std')
            plt.axvline(x=best_k_cv, color='red', linestyle='--', alpha=0.7, 
                       label=f'Best K={best_k_cv}')
            plt.xlabel('Number of Neighbors (K)')
            plt.ylabel('Cross-Validation Accuracy')
            plt.title('KNN Cross-Validation Results')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            cv_filename = os.path.join(output_dir, f'knn_cross_validation_{timestamp}.pdf')
            plt.savefig(cv_filename, format='pdf', dpi=300, bbox_inches='tight')
            plot_files.append(cv_filename)
            plt.close()
            
            # Test accuracy plot
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, test_acc, 'g-', linewidth=2, label='Test Set Accuracy')
            plt.axvline(x=best_k_test, color='red', linestyle='--', alpha=0.7, 
                       label=f'Best K={best_k_test}')
            plt.xlabel('Number of Neighbors (K)')
            plt.ylabel('Test Set Accuracy')
            plt.title('KNN Test Set Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            test_filename = os.path.join(output_dir, f'knn_test_accuracy_{timestamp}.pdf')
            plt.savefig(test_filename, format='pdf', dpi=300, bbox_inches='tight')
            plot_files.append(test_filename)
            plt.close()
            
            # Error rate plot
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, error_rate, color='blue', linestyle='dashed', 
                    marker='o', markerfacecolor='red', markersize=8, linewidth=2, 
                    label='Error Rate')
            plt.axvline(x=best_k_error, color='red', linestyle='--', alpha=0.7, 
                       label=f'Min Error K={best_k_error}')
            plt.xlabel('Number of Neighbors (K)')
            plt.ylabel('Error Rate')
            plt.title('KNN Error Rate vs K Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            error_filename = os.path.join(output_dir, f'knn_error_rate_{timestamp}.pdf')
            plt.savefig(error_filename, format='pdf', dpi=300, bbox_inches='tight')
            plot_files.append(error_filename)
            plt.close()
            
            print(f"Individual plots saved as:")
            print(f"  - {cv_filename}")
            print(f"  - {test_filename}")
            print(f"  - {error_filename}")
    
    # Print results
    print(f"\n=== Cross-Validation Results ===")
    print(f"Best CV accuracy: {best_acc_cv:.4f} (±{std_acc[best_cv_idx]:.4f}) with K={best_k_cv}")
    print(f"\n=== Test Set Results ===")
    print(f"Best test accuracy: {best_acc_test:.4f} with K={best_k_test}")
    print(f"\n=== Error Rate Results ===")
    print(f"Minimum error rate: {min_error:.4f} with K={best_k_error}")
    
    # Check if all methods agree on the best K
    if best_k_cv == best_k_test == best_k_error:
        print(f"\n🎯 All methods agree! Recommended K = {best_k_cv}")
    else:
        print(f"\n📊 Different methods suggest different K values:")
        print(f"   - Cross-validation recommends K = {best_k_cv}")
        print(f"   - Test accuracy recommends K = {best_k_test}")  
        print(f"   - Error rate recommends K = {best_k_error}")
        print(f"   - Cross-validation is generally more reliable for model selection")
    
    # Return comprehensive results
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
        'plot_files': plot_files if save_plots else []
    }

# Example usage:
"""
# Basic usage - display plots only
results = tune_knn_hyperparameters(X_train, y_train, X_test, y_test, max_k=30)

# Save plots as PDFs in current directory
results = tune_knn_hyperparameters(X_train, y_train, X_test, y_test, 
                                 max_k=30, save_plots=True)

# Save plots in specific directory
results = tune_knn_hyperparameters(X_train, y_train, X_test, y_test, 
                                 max_k=30, save_plots=True, 
                                 output_dir='./knn_results/')

# Access results:
print(f"Recommended K value: {results['best_k']}")
print(f"Expected accuracy: {results['best_accuracy']:.4f}")
print(f"Saved plots: {results['plot_files']}")

# Get the accuracies for all K values:
k_vals = results['k_values']
cv_accs = results['mean_accuracies']
test_accs = results['test_accuracies']
error_rates = results['error_rates']
"""