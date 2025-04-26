import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def ensure_static_dir():
    """Ensure the static directory exists."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    os.makedirs(static_dir, exist_ok=True)
    return static_dir

def get_plotly_theme_config():
    """Get common Plotly theme configuration."""
    return {
        'template': 'plotly_white',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': dict(size=12),
        'margin': dict(l=60, r=30, t=50, b=50),
        'xaxis': dict(
            gridcolor='rgba(128, 128, 128, 0.1)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        ),
        'yaxis': dict(
            gridcolor='rgba(128, 128, 128, 0.1)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        ),
        'legend': dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(128, 128, 128, 0.2)',
            borderwidth=1
        )
    }

def generate_confusion_matrices():
    """Generate confusion matrices for each fold."""
    static_dir = ensure_static_dir()
    
    # Sample confusion matrix data for each fold
    fold_data = [
        np.array([[450, 50], [30, 470]]),  # Fold 1
        np.array([[460, 40], [35, 465]]),  # Fold 2
        np.array([[455, 45], [25, 475]]),  # Fold 3
        np.array([[465, 35], [40, 460]]),  # Fold 4
        np.array([[470, 30], [20, 480]]),  # Fold 5
    ]
    
    for theme in ['light', 'dark']:
        # Set theme-specific colors
        if theme == 'dark':
            text_color = 'white'
            bg_color = '#1a1a1a'
            cmap = 'YlOrRd'
            plt.style.use('dark_background')
        else:
            text_color = 'black'
            bg_color = 'white'
            cmap = 'Blues'
            plt.style.use('default')
        
        for i, cm in enumerate(fold_data, 1):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_facecolor(bg_color)
            plt.gcf().set_facecolor(bg_color)
            
            # Create heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                cmap=cmap,
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            
            # Set labels manually
            ax.set_xticklabels(['Normal', 'Abnormal'])
            ax.set_yticklabels(['Normal', 'Abnormal'])
            
            # Customize text colors
            plt.title(f'Confusion Matrix - Fold {i}', color=text_color)
            plt.ylabel('True Label', color=text_color)
            plt.xlabel('Predicted Label', color=text_color)
            
            # Update tick colors
            ax.tick_params(colors=text_color)
            
            plt.tight_layout()
            theme_suffix = '_dark' if theme == 'dark' else ''
            plt.savefig(
                os.path.join(static_dir, f'confusion_matrix_fold_{i}{theme_suffix}.png'),
                facecolor=bg_color,
                edgecolor='none',
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

def generate_roc_curves(fold_histories):
    """Generate interactive ROC curves for each fold."""
    static_dir = ensure_static_dir()
    
    for i, fold_history in enumerate(fold_histories, 1):
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(fold_history['true_labels'], fold_history['pred_probs'])
        auc_score = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr, 
            name=f'ROC curve (AUC = {auc_score:.3f})',
            line=dict(color='royalblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1], 
            name='Random',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - Fold {i}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            width=600,
            height=400,
            **get_plotly_theme_config()
        )
        
        fig.write_html(
            os.path.join(static_dir, f'roc_curve_fold_{i}.html'),
            include_plotlyjs='cdn',
            full_html=False,
            config={'responsive': True}
        )

def generate_pr_curves(fold_histories):
    """Generate interactive precision-recall curves for each fold."""
    static_dir = ensure_static_dir()
    
    for i, fold_history in enumerate(fold_histories, 1):
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(fold_history['true_labels'], fold_history['pred_probs'])
        ap_score = average_precision_score(fold_history['true_labels'], fold_history['pred_probs'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, 
            y=precision, 
            name=f'PR curve (AP = {ap_score:.3f})',
            line=dict(color='royalblue', width=2)
        ))
        
        fig.update_layout(
            title=f'Precision-Recall Curve - Fold {i}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            showlegend=True,
            width=600,
            height=400,
            **get_plotly_theme_config()
        )
        
        fig.write_html(
            os.path.join(static_dir, f'pr_curve_fold_{i}.html'),
            include_plotlyjs='cdn',
            full_html=False,
            config={'responsive': True}
        )

def generate_training_history(history):
    """Generate interactive training history plot."""
    static_dir = ensure_static_dir()
    
    # Convert range to list if needed
    epochs = list(range(1, len(history['train_acc']) + 1))
    
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=('Model Accuracy', 'Model Loss'),
        vertical_spacing=0.15
    )
    
    # Accuracy subplot
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['train_acc'], 
            name='Train Accuracy',
            line=dict(color='royalblue', width=2)
        ), 
        row=1, 
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['val_acc'], 
            name='Validation Accuracy',
            line=dict(color='coral', width=2)
        ), 
        row=1, 
        col=1
    )
    
    # Loss subplot
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['train_loss'], 
            name='Train Loss',
            line=dict(color='royalblue', width=2)
        ), 
        row=2, 
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=history['val_loss'], 
            name='Validation Loss',
            line=dict(color='coral', width=2)
        ), 
        row=2, 
        col=1
    )
    
    theme_config = get_plotly_theme_config()
    fig.update_layout(
        height=800,
        showlegend=True,
        **theme_config
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, gridcolor=theme_config['xaxis']['gridcolor'])
    fig.update_yaxes(title_text="Loss", row=2, col=1, gridcolor=theme_config['xaxis']['gridcolor'])
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Epoch", row=2, col=1, gridcolor=theme_config['xaxis']['gridcolor'])
    
    fig.write_html(
        os.path.join(static_dir, 'training_history_interactive.html'),
        include_plotlyjs='cdn',
        full_html=False,
        config={'responsive': True}
    )

def generate_lr_schedule(history):
    """Generate interactive learning rate schedule plot."""
    static_dir = ensure_static_dir()
    
    # Convert range to list if needed
    epochs = list(range(1, len(history['learning_rates']) + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=history['learning_rates'], 
        name='Learning Rate',
        line=dict(color='royalblue', width=2)
    ))
    
    fig.update_layout(
        title='Learning Rate Schedule',
        xaxis_title='Epoch',
        yaxis_title='Learning Rate',
        showlegend=True,
        width=800,
        height=400,
        yaxis_type='log',
        **get_plotly_theme_config()
    )
    
    fig.write_html(
        os.path.join(static_dir, 'lr_schedule_interactive.html'),
        include_plotlyjs='cdn',
        full_html=False,
        config={'responsive': True}
    )

def main():
    """Generate all visualization files for testing/development."""
    print("Generating visualization files...")
    
    # Generate confusion matrices
    generate_confusion_matrices()
    print("✓ Generated confusion matrices")
    
    # Create sample data for testing
    sample_fold_histories = []
    for i in range(5):
        n_samples = 1000
        true_labels = np.random.randint(0, 2, n_samples)
        pred_probs = np.clip(np.random.normal(true_labels, 0.2), 0, 1)
        sample_fold_histories.append({
            'true_labels': true_labels,
            'predictions': (pred_probs > 0.5).astype(int),
            'pred_probs': pred_probs
        })
    
    # Create sample training history
    sample_history = {
        'train_acc': [0.7 + 0.02 * i for i in range(50)],
        'val_acc': [0.65 + 0.02 * i + np.random.normal(0, 0.01) for i in range(50)],
        'train_loss': [0.8 * np.exp(-0.05 * i) for i in range(50)],
        'val_loss': [0.9 * np.exp(-0.05 * i) + 0.1 for i in range(50)],
        'learning_rates': [0.001 * np.exp(-0.02 * i) for i in range(50)]
    }
    
    # Generate visualizations with sample data
    generate_roc_curves(sample_fold_histories)
    print("✓ Generated ROC curves")
    
    generate_pr_curves(sample_fold_histories)
    print("✓ Generated PR curves")
    
    generate_training_history(sample_history)
    print("✓ Generated training history")
    
    generate_lr_schedule(sample_history)
    print("✓ Generated learning rate schedule")
    
    print("\nAll visualization files have been generated successfully!")

if __name__ == '__main__':
    main() 