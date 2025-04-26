import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from datetime import datetime

def load_training_data(history_path, metrics_path):
    """Load training history and test metrics data from JSON files"""
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        with open(metrics_path, 'r') as f:
            test_metrics = json.load(f)
        
        return history, test_metrics
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def plot_training_metrics(history, output_dir, show_validation=True, figsize=(16, 12), dpi=300):
    """
    Create a comprehensive plot of all training metrics
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save the plots
        show_validation: Whether to show validation metrics
        figsize: Figure size
        dpi: Resolution for saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # Create a 2x1 subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Get epochs
    epochs = history.get('epoch', list(range(1, len(history['train_loss']) + 1)))
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    if show_validation and 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    
    # Customize Loss plot
    ax1.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    if show_validation and 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    
    # Customize Accuracy plot
    ax2.set_title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Format y-axis as percentage for accuracy
    ax2.set_ylim([0, 1])
    ax2.set_yticklabels([f'{x:.0%}' for x in ax2.get_yticks()])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'enhanced_training_metrics_{timestamp}.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced training metrics plot saved to: {output_path}")
    return output_path

def plot_metrics_comparison(history, test_metrics, output_dir, figsize=(20, 16), dpi=300):
    """
    Create a comprehensive dashboard of all metrics
    
    Args:
        history: Dictionary containing training history
        test_metrics: Dictionary containing test metrics
        output_dir: Directory to save the plots
        figsize: Figure size
        dpi: Resolution for saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get epochs
    epochs = history.get('epoch', list(range(1, len(history['train_loss']) + 1)))
    
    # 1. Top Left: Training & Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 2. Top Right: Training & Validation Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    if 'val_acc' in history:
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_yticklabels([f'{x:.0%}' for x in axes[0, 1].get_yticks()])
    
    # 3. Bottom Left: Confusion Matrix
    if 'confusion_matrix' in test_metrics:
        cm = np.array(test_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
        axes[1, 0].set_ylabel('True Label', fontsize=12)
        
        # Determine which labels to use based on matrix size
        if cm.shape == (2, 2):
            axes[1, 0].set_xticklabels(['Negative', 'Positive'])
            axes[1, 0].set_yticklabels(['Negative', 'Positive'])
    else:
        axes[1, 0].text(0.5, 0.5, 'Confusion Matrix not available', 
                      horizontalalignment='center', verticalalignment='center',
                      fontsize=14, transform=axes[1, 0].transAxes)
    
    # 4. Bottom Right: Test Metrics Summary
    important_metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc']
    available_metrics = [m for m in important_metrics if m in test_metrics]
    
    if available_metrics:
        axes[1, 1].axis('off')
        text_content = "Test Metrics Summary:\n\n"
        
        for metric in available_metrics:
            value = test_metrics[metric]
            formatted_value = f"{value:.4f}"
            text_content += f"{metric.replace('_', ' ').title()}: {formatted_value}\n"
            
        axes[1, 1].text(0.1, 0.9, text_content, 
                       horizontalalignment='left', verticalalignment='top',
                       fontsize=14, transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, 'Test Metrics not available', 
                      horizontalalignment='center', verticalalignment='center',
                      fontsize=14, transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'metrics_dashboard_{timestamp}.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive metrics dashboard saved to: {output_path}")
    return output_path

def plot_roc_and_pr_curves(test_metrics, output_dir, figsize=(16, 8), dpi=300):
    """
    Plot ROC and Precision-Recall curves
    
    Args:
        test_metrics: Dictionary containing test metrics
        output_dir: Directory to save the plots
        figsize: Figure size
        dpi: Resolution for saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set style
    sns.set_style("whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. ROC Curve
    if 'roc_curve' in test_metrics and isinstance(test_metrics['roc_curve'], dict):
        roc_data = test_metrics['roc_curve']
        if all(k in roc_data for k in ['fpr', 'tpr']):
            fpr = np.array(roc_data['fpr'])
            tpr = np.array(roc_data['tpr'])
            roc_auc = test_metrics.get('roc_auc', None)
            
            ax1.plot(fpr, tpr, 'b-', linewidth=2, 
                   label=f'ROC Curve (AUC = {roc_auc:.4f})' if roc_auc is not None else 'ROC Curve')
            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax1.set_title('ROC Curve', fontsize=16, fontweight='bold')
            ax1.set_xlabel('False Positive Rate', fontsize=12)
            ax1.set_ylabel('True Positive Rate', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'ROC data missing keys', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=14, transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, 'ROC Curve data not available', 
               horizontalalignment='center', verticalalignment='center',
               fontsize=14, transform=ax1.transAxes)
    
    # 2. Precision-Recall Curve
    if 'pr_curve' in test_metrics and isinstance(test_metrics['pr_curve'], dict):
        pr_data = test_metrics['pr_curve']
        if all(k in pr_data for k in ['precision', 'recall']):
            precision = np.array(pr_data['precision'])
            recall = np.array(pr_data['recall'])
            pr_auc = test_metrics.get('pr_auc', None)
            
            ax2.plot(recall, precision, 'r-', linewidth=2, 
                   label=f'PR Curve (AUC = {pr_auc:.4f})' if pr_auc is not None else 'PR Curve')
            ax2.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Recall', fontsize=12)
            ax2.set_ylabel('Precision', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'PR data missing keys', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=14, transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Precision-Recall Curve data not available', 
               horizontalalignment='center', verticalalignment='center',
               fontsize=14, transform=ax2.transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'roc_pr_curves_{timestamp}.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"ROC and PR curves saved to: {output_path}")
    return output_path

def generate_advanced_visualizations(history_path, metrics_path, output_dir):
    """
    Generate all advanced visualization plots
    
    Args:
        history_path: Path to training history JSON file
        metrics_path: Path to test metrics JSON file
        output_dir: Directory to save visualizations
    """
    print("\n========== GENERATING ADVANCED VISUALIZATION PLOTS ==========")
    
    try:
        # Load data
        history, test_metrics = load_training_data(history_path, metrics_path)
        if not history or not test_metrics:
            return False
            
        # Create visualizations directory if it doesn't exist
        vis_dir = os.path.join(output_dir, 'advanced_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate plots
        plot_training_metrics(history, vis_dir)
        plot_metrics_comparison(history, test_metrics, vis_dir)
        plot_roc_and_pr_curves(test_metrics, vis_dir)
        
        print("\nAll advanced visualization plots generated successfully!")
        print(f"Plots saved to: {vis_dir}")
        return True
        
    except Exception as e:
        print(f"Error generating advanced visualization plots: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python advanced_visualizations.py <history_file> <metrics_file> [output_dir]")
        sys.exit(1)
    
    history_path = sys.argv[1]
    metrics_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(history_path)
    
    generate_advanced_visualizations(history_path, metrics_path, output_dir) 