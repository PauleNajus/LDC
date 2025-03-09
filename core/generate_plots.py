import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

def find_latest_files(directory):
    """Find the latest training history and test metrics files"""
    history_files = [f for f in os.listdir(directory) if f.startswith('training_history_') and f.endswith('.json')]
    test_metrics_files = [f for f in os.listdir(directory) if f.startswith('test_metrics_') and f.endswith('.json')]
    
    if not history_files or not test_metrics_files:
        print(f"No history or metrics files found in {directory}")
        print(f"Files in directory: {os.listdir(directory)}")
        return None, None
    
    # Sort by timestamp (assuming format: filename_YYYYMMDD_HHMMSS.json)
    history_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    test_metrics_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    
    history_path = os.path.join(directory, history_files[0])
    metrics_path = os.path.join(directory, test_metrics_files[0])
    
    print(f"Found latest history file: {history_files[0]}")
    print(f"Found latest metrics file: {test_metrics_files[0]}")
    
    return history_path, metrics_path

def load_training_data(history_path, metrics_path):
    """Load training history and test metrics from JSON files"""
    history = None
    test_metrics = None
    
    if not os.path.exists(history_path) or not os.path.exists(metrics_path):
        print(f"Error: Could not find history file {history_path} or metrics file {metrics_path}")
        return None, None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error loading training history: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading training history: {str(e)}")
        return None, None
    
    try:
        with open(metrics_path, 'r') as f:
            test_metrics = json.load(f)
            
            # Convert confusion matrix to numpy array if it exists
            if 'confusion_matrix' in test_metrics and isinstance(test_metrics['confusion_matrix'], list):
                test_metrics['confusion_matrix'] = np.array(test_metrics['confusion_matrix'])
                
            # Ensure ROC curve data is properly formatted
            if 'roc_curve' in test_metrics:
                if isinstance(test_metrics['roc_curve'], dict):
                    # Ensure all components are lists or arrays
                    for key in ['fpr', 'tpr', 'thresholds']:
                        if key in test_metrics['roc_curve'] and not isinstance(test_metrics['roc_curve'][key], (list, np.ndarray)):
                            test_metrics['roc_curve'][key] = [test_metrics['roc_curve'][key]]
                else:
                    # Remove invalid ROC curve data
                    print("Warning: ROC curve data is not correctly formatted, skipping ROC analysis")
                    test_metrics.pop('roc_curve', None)
                
    except json.JSONDecodeError as e:
        print(f"Error loading test metrics: {str(e)}")
        return history, None
    except Exception as e:
        print(f"Unexpected error loading test metrics: {str(e)}")
        return history, None
    
    return history, test_metrics

def plot_training_vs_validation_curves(history, output_dir):
    """
    Plot training vs validation loss and accuracy curves
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training and validation loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'train_val_curves_{timestamp}.png'), dpi=300)
    plt.close()
    
    print(f"Training and validation curves saved to {output_dir}")

def plot_confusion_matrix_heatmap(test_metrics, output_dir):
    """
    Create a heatmap of the confusion matrix
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'confusion_matrix' not in test_metrics:
        print("Confusion matrix data not found in test metrics")
        return
    
    cm = test_metrics['confusion_matrix']
    if isinstance(cm, list):
        cm = np.array(cm)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    
    # Add labels and title
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    # Set tick labels
    ax.set_xticklabels(['Normal', 'Pneumonia'])
    ax.set_yticklabels(['Normal', 'Pneumonia'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_heatmap_{timestamp}.png'), dpi=300)
    plt.close()
    
    print(f"Confusion matrix heatmap saved to {output_dir}")

def plot_roc_curve(test_metrics, output_dir):
    """
    Plot ROC curve and AUC
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'roc_curve' not in test_metrics or 'roc_auc' not in test_metrics:
        print("ROC curve data not found in test metrics")
        return
    
    plt.figure(figsize=(10, 8))
    
    fpr = test_metrics['roc_curve']['fpr']
    tpr = test_metrics['roc_curve']['tpr']
    roc_auc = test_metrics['roc_auc']
    
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_curve_{timestamp}.png'), dpi=300)
    plt.close()
    
    print(f"ROC curve saved to {output_dir}")

def plot_precision_recall_curve(test_metrics, output_dir):
    """
    Plot Precision-Recall curve
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'pr_curve' not in test_metrics or 'pr_auc' not in test_metrics:
        print("Precision-Recall curve data not found in test metrics")
        return
    
    plt.figure(figsize=(10, 8))
    
    precision = test_metrics['pr_curve']['precision']
    recall = test_metrics['pr_curve']['recall']
    pr_auc = test_metrics['pr_auc']
    
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'precision_recall_curve_{timestamp}.png'), dpi=300)
    plt.close()
    
    print(f"Precision-Recall curve saved to {output_dir}")

def plot_learning_rate_schedule(history, output_dir):
    """
    Plot learning rate schedule over epochs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'lr' not in history:
        print("Learning rate data not found in training history")
        return
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['lr']) + 1)
    plt.plot(epochs, history['lr'], 'b-', linewidth=2)
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.grid(True)
    
    # Use scientific notation for y-axis if learning rates are very small
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'learning_rate_schedule_{timestamp}.png'), dpi=300)
    plt.close()
    
    print(f"Learning rate schedule plot saved to {output_dir}")

def plot_threshold_analysis(test_metrics, output_dir):
    """
    Plot threshold analysis showing how different metrics vary with threshold
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'roc_curve' not in test_metrics:
        print("ROC curve data not found for threshold analysis")
        return
    
    # Print debug info about ROC curve data
    print("ROC curve data types:")
    for key, value in test_metrics['roc_curve'].items():
        print(f"  {key}: {type(value)}")
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            print(f"    First element type: {type(value[0])}")
    
    # Extract data from test metrics
    try:
        fpr = np.array(test_metrics['roc_curve']['fpr'])
        tpr = np.array(test_metrics['roc_curve']['tpr'])
        thresholds = np.array(test_metrics['roc_curve']['thresholds'])
        
        # Calculate specificity (1 - fpr)
        specificity = 1 - fpr
        
        # Calculate balanced accuracy
        balanced_acc = (tpr + specificity) / 2
        
        # Create a DataFrame for plotting
        threshold_df = pd.DataFrame({
            'Threshold': thresholds,
            'Sensitivity': tpr,
            'Specificity': specificity,
            'Balanced Accuracy': balanced_acc
        })
        
        # Filter out extreme threshold values for better visualization
        threshold_df = threshold_df[(threshold_df['Threshold'] >= 0.01) & (threshold_df['Threshold'] <= 0.99)]
        
        plt.figure(figsize=(12, 8))
        
        # Plot each metric against threshold
        plt.plot(threshold_df['Threshold'], threshold_df['Sensitivity'], 'b-', linewidth=2, label='Sensitivity')
        plt.plot(threshold_df['Threshold'], threshold_df['Specificity'], 'r-', linewidth=2, label='Specificity')
        plt.plot(threshold_df['Threshold'], threshold_df['Balanced Accuracy'], 'g-', linewidth=2, label='Balanced Accuracy')
        
        # Add vertical line at threshold = 0.5
        plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Threshold = 0.5')
        
        # Highlight the threshold with highest balanced accuracy
        if not threshold_df.empty:
            best_idx = threshold_df['Balanced Accuracy'].idxmax()
            best_threshold = threshold_df.loc[best_idx, 'Threshold']
            best_balanced_acc = threshold_df.loc[best_idx, 'Balanced Accuracy']
            
            print(f"Best threshold: {best_threshold} (type: {type(best_threshold)})")
            print(f"Best balanced accuracy: {best_balanced_acc} (type: {type(best_balanced_acc)})")
            
            # Ensure values are of the right type
            if isinstance(best_threshold, (int, float)) and isinstance(best_balanced_acc, (int, float)):
                plt.scatter(best_threshold, best_balanced_acc, color='black', s=100, zorder=5)
                plt.annotate(f'Best Threshold: {best_threshold:.3f}',
                            xy=(best_threshold, best_balanced_acc),
                            xytext=(best_threshold + 0.1, best_balanced_acc),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=10)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Classification Threshold', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.title('Threshold Analysis for Classification Metrics', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'threshold_analysis_{timestamp}.png'), dpi=300)
        plt.close()
        
        print(f"Threshold analysis plot saved to {output_dir}")
    except Exception as e:
        print(f"Error in threshold analysis plot: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Skipping threshold analysis plot")

def generate_all_plots():
    """Generate all visualization plots"""
    # Find directories for model data
    models_dir = os.path.join(project_root, 'models')
    train_logs_dir = os.path.join(project_root, 'train_results_logs')
    output_dir = os.path.join(project_root, 'train_results_logs', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find latest training history and test metrics files
    history_path, metrics_path = find_latest_files(train_logs_dir)
    
    if not history_path or not metrics_path:
        print("Could not find training history or test metrics files")
        return
    
    print(f"Using training history: {history_path}")
    print(f"Using test metrics: {metrics_path}")
    
    # Load data
    history, test_metrics = load_training_data(history_path, metrics_path)
    
    if not history or not test_metrics:
        print("Failed to load training data")
        return
    
    print("Generating plots...")
    
    # Print structure of test_metrics
    print("Test metrics keys:", list(test_metrics.keys()))
    if 'roc_curve' in test_metrics:
        print("ROC curve keys:", list(test_metrics['roc_curve'].keys()))
    
    try:
        # Generate all requested plots
        plot_training_vs_validation_curves(history, output_dir)
        plot_confusion_matrix_heatmap(test_metrics, output_dir)
        plot_roc_curve(test_metrics, output_dir)
        plot_precision_recall_curve(test_metrics, output_dir)
        plot_learning_rate_schedule(history, output_dir)
        plot_threshold_analysis(test_metrics, output_dir)
        
        print("\nAll plots generated successfully!")
        print(f"Plots saved to: {output_dir}")
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set the style for all plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Generate all plots
    generate_all_plots() 