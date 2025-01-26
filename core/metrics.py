"""Module for storing model metrics and performance data."""

# Cross-validation metrics
CV_METRICS = {
    'cv_avg_accuracy': 92.5,  # Average accuracy across 5 folds
    'cv_avg_precision': 91.8,  # Average precision across 5 folds
    'cv_avg_recall': 93.2,  # Average recall across 5 folds
    'cv_avg_f1': 92.5,  # Average F1 score across 5 folds
    'best_val_accuracy': 94.3,  # Best validation accuracy achieved
    'total_training_time': 3.5,  # Total training time in hours
    'gpu_memory_used_gb': 8.2,  # GPU memory used in GB
    'convergence_epoch': 45,  # Epoch where model converged
} 