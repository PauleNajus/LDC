import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            ax.set_xticklabels(['Normal', 'Pneumonia'])
            ax.set_yticklabels(['Normal', 'Pneumonia'])
            
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

def generate_roc_curves():
    """Generate interactive ROC curves for each fold."""
    static_dir = ensure_static_dir()
    
    # Sample ROC curve data for each fold
    fpr_tpr_data = [
        (np.linspace(0, 1, 100), np.power(np.linspace(0, 1, 100), 0.3)),  # Fold 1
        (np.linspace(0, 1, 100), np.power(np.linspace(0, 1, 100), 0.25)),  # Fold 2
        (np.linspace(0, 1, 100), np.power(np.linspace(0, 1, 100), 0.28)),  # Fold 3
        (np.linspace(0, 1, 100), np.power(np.linspace(0, 1, 100), 0.27)),  # Fold 4
        (np.linspace(0, 1, 100), np.power(np.linspace(0, 1, 100), 0.26)),  # Fold 5
    ]
    
    for i, (fpr, tpr) in enumerate(fpr_tpr_data, 1):
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr, 
            name=f'ROC curve (AUC = {auc:.3f})',
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

def generate_pr_curves():
    """Generate interactive precision-recall curves for each fold."""
    static_dir = ensure_static_dir()
    
    # Sample precision-recall data for each fold
    prec_recall_data = [
        (np.linspace(1, 0, 100), np.power(np.linspace(1, 0, 100), 2)),  # Fold 1
        (np.linspace(1, 0, 100), np.power(np.linspace(1, 0, 100), 1.9)),  # Fold 2
        (np.linspace(1, 0, 100), np.power(np.linspace(1, 0, 100), 1.95)),  # Fold 3
        (np.linspace(1, 0, 100), np.power(np.linspace(1, 0, 100), 1.85)),  # Fold 4
        (np.linspace(1, 0, 100), np.power(np.linspace(1, 0, 100), 1.92)),  # Fold 5
    ]
    
    for i, (precision, recall) in enumerate(prec_recall_data, 1):
        # Calculate average precision
        ap = np.trapz(precision, recall)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, 
            y=precision, 
            name=f'PR curve (AP = {ap:.3f})',
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

def generate_training_history():
    """Generate interactive training history plot."""
    static_dir = ensure_static_dir()
    
    # Sample training history data
    epochs = np.arange(1, 51)
    train_acc = 1 / (1 + np.exp(-0.2 * (epochs - 10))) * 0.3 + 0.65
    val_acc = train_acc + np.random.normal(0, 0.02, len(epochs))
    train_loss = 0.8 * np.exp(-0.1 * epochs) + 0.2
    val_loss = train_loss + np.random.normal(0, 0.05, len(epochs))
    
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
            y=train_acc, 
            name='Train Accuracy',
            line=dict(color='royalblue', width=2)
        ), 
        row=1, 
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=val_acc, 
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
            y=train_loss, 
            name='Train Loss',
            line=dict(color='royalblue', width=2)
        ), 
        row=2, 
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, 
            y=val_loss, 
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

def generate_lr_schedule():
    """Generate interactive learning rate schedule plot."""
    static_dir = ensure_static_dir()
    
    # Sample learning rate schedule data
    epochs = np.arange(1, 51)
    lr = 0.001 * np.exp(-0.05 * epochs)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=lr, 
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
    """Generate all visualization files."""
    print("Generating visualization files...")
    
    # Generate all visualizations
    generate_confusion_matrices()
    print("✓ Generated confusion matrices")
    
    generate_roc_curves()
    print("✓ Generated ROC curves")
    
    generate_pr_curves()
    print("✓ Generated PR curves")
    
    generate_training_history()
    print("✓ Generated training history")
    
    generate_lr_schedule()
    print("✓ Generated learning rate schedule")
    
    print("\nAll visualization files have been generated successfully!")

if __name__ == "__main__":
    main() 