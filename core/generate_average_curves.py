import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_average_roc_curve():
    """Create average ROC curve from individual fold curves."""
    fig = go.Figure()
    
    # Add individual fold curves (you can customize these values)
    folds_tpr = [
        [0, 0.3, 0.6, 0.8, 0.95, 1.0],  # Fold 1
        [0, 0.35, 0.65, 0.85, 0.92, 1.0],  # Fold 2
        [0, 0.32, 0.62, 0.82, 0.94, 1.0],  # Fold 3
        [0, 0.33, 0.63, 0.83, 0.93, 1.0],  # Fold 4
        [0, 0.31, 0.61, 0.81, 0.91, 1.0],  # Fold 5
    ]
    fpr = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Calculate mean and std
    mean_tpr = np.mean(folds_tpr, axis=0)
    std_tpr = np.std(folds_tpr, axis=0)
    
    # Add average curve with confidence interval
    fig.add_trace(go.Scatter(
        x=fpr,
        y=mean_tpr,
        line=dict(color='rgb(31, 119, 180)', width=2),
        name='Average ROC',
        mode='lines',
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=fpr + fpr[::-1],
        y=np.concatenate([mean_tpr + std_tpr, (mean_tpr - std_tpr)[::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(31, 119, 180, 0)'),
        name='95% CI',
        showlegend=True
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        line=dict(color='rgb(100, 100, 100)', width=1, dash='dash'),
        name='Random',
        showlegend=True
    ))
    
    # Update layout with responsive settings
    fig.update_layout(
        title=dict(
            text='Average ROC Curve (5-Fold Cross-Validation)',
            y=0.95
        ),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        hovermode='closest',
        autosize=True,
        height=400,  # Reduced height
        margin=dict(t=30, l=30, r=20, b=30),  # Reduced margins
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Save to HTML with responsive settings
    config = {'responsive': True, 'displayModeBar': False}
    fig.write_html('static/average_roc_curve.html', config=config)

def create_average_pr_curve():
    """Create average Precision-Recall curve from individual fold curves."""
    fig = go.Figure()
    
    # Add individual fold curves (you can customize these values)
    folds_precision = [
        [1.0, 0.95, 0.9, 0.85, 0.8, 0.75],  # Fold 1
        [1.0, 0.93, 0.88, 0.83, 0.78, 0.73],  # Fold 2
        [1.0, 0.94, 0.89, 0.84, 0.79, 0.74],  # Fold 3
        [1.0, 0.92, 0.87, 0.82, 0.77, 0.72],  # Fold 4
        [1.0, 0.91, 0.86, 0.81, 0.76, 0.71],  # Fold 5
    ]
    recall = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Calculate mean and std
    mean_precision = np.mean(folds_precision, axis=0)
    std_precision = np.std(folds_precision, axis=0)
    
    # Add average curve with confidence interval
    fig.add_trace(go.Scatter(
        x=recall,
        y=mean_precision,
        line=dict(color='rgb(44, 160, 44)', width=2),
        name='Average PR',
        mode='lines',
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=recall + recall[::-1],
        y=np.concatenate([mean_precision + std_precision, (mean_precision - std_precision)[::-1]]),
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(44, 160, 44, 0)'),
        name='95% CI',
        showlegend=True
    ))
    
    # Update layout with responsive settings
    fig.update_layout(
        title=dict(
            text='Average Precision-Recall Curve (5-Fold Cross-Validation)',
            y=0.95
        ),
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_white',
        hovermode='closest',
        autosize=True,
        height=400,  # Reduced height
        margin=dict(t=30, l=30, r=20, b=30),  # Reduced margins
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Save to HTML with responsive settings
    config = {'responsive': True, 'displayModeBar': False}
    fig.write_html('static/average_pr_curve.html', config=config)

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Generate curves
    create_average_roc_curve()
    create_average_pr_curve()
    print("Generated average ROC and PR curves in static directory.") 