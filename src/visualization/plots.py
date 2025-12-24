import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_error_distribution(errors, labels, threshold, save_path=None):
    """Plots the distribution of reconstruction errors for Normal vs Anomaly."""
    plt.figure(figsize=(10, 6))
    
    # Healthy
    sns.histplot(errors[labels == 0], color="blue", label="Healthy", kde=True, stat="density", bins=50)
    
    # Unhealthy (if any exist in the provided data)
    if np.sum(labels == 1) > 0:
        sns.histplot(errors[labels == 1], color="red", label="Anomaly", kde=True, stat="density", bins=50)
        
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.title('Reconstruction Error Distribution', fontsize=16)
    plt.xlabel('Reconstruction Error (MAE)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_reconstruction(original, reconstructed, error, threshold, title="Reconstruction", save_path=None):
    """Plots a single window comparison."""
    plt.figure(figsize=(12, 4))
    plt.plot(original, label='Original', color='blue')
    plt.plot(reconstructed, label='Reconstructed', color='red', linestyle='--')
    
    status = "Anomaly" if error > threshold else "Normal"
    plt.title(f"{title} | Error: {error:.4f} | Status: {status}")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plots a heatmap of the confusion matrix."""
    import pandas as pd
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(7, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 16})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()