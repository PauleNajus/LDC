import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

def create_average_roc_curve(fold_histories):
    """Create average ROC curve from individual fold curves."""
    fig = go.Figure()
    
    # Calculate ROC curves for each fold
    all_fprs = []
    all_tprs = []
    for fold_history in fold_histories:
        fpr, tpr, _ = roc_curve(fold_history['true_labels'], fold_history['pred_probs'])
        all_fprs.append(fpr)
        all_tprs.append(tpr)
    
    # Interpolate all ROC curves to a common set of FPR points
    mean_fpr = np.array(np.linspace(0, 1, 100))
    interp_tprs = []
    for fpr, tpr in zip(all_fprs, all_tprs):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    
    # Calculate mean and std of TPR
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interp_tprs, axis=0)
    
    # Calculate mean AUC
    mean_auc = auc(mean_fpr, mean_tpr)
    
    # Add average curve
    fig.add_trace(go.Scatter(
        x=mean_fpr.tolist(),
        y=mean_tpr.tolist(),
        line=dict(color='rgb(31, 119, 180)', width=2),
        name=f'Average ROC (AUC = {mean_auc:.3f})',
        mode='lines',
    ))
    
    # Add confidence interval
    ci_x = np.concatenate([mean_fpr, mean_fpr[::-1]]).tolist()
    ci_y = np.concatenate([mean_tpr + std_tpr, (mean_tpr - std_tpr)[::-1]]).tolist()
    fig.add_trace(go.Scatter(
        x=ci_x,
        y=ci_y,
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

def create_average_pr_curve(fold_histories):
    """Create average Precision-Recall curve from individual fold curves."""
    fig = go.Figure()
    
    # Calculate PR curves for each fold
    all_precisions = []
    all_recalls = []
    for fold_history in fold_histories:
        precision, recall, _ = precision_recall_curve(fold_history['true_labels'], fold_history['pred_probs'])
        all_precisions.append(precision)
        all_recalls.append(recall)
    
    # Interpolate all PR curves to a common set of recall points
    mean_recall = np.array(np.linspace(0, 1, 100))
    interp_precisions = []
    for recall, precision in zip(all_recalls, all_precisions):
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        interp_precisions.append(interp_precision)
    
    # Calculate mean and std of precision
    mean_precision = np.mean(interp_precisions, axis=0)
    std_precision = np.std(interp_precisions, axis=0)
    
    # Calculate mean average precision
    mean_ap = np.mean(np.array([
        average_precision_score(fold_history['true_labels'], fold_history['pred_probs'])
        for fold_history in fold_histories
    ]))
    
    # Add average curve
    fig.add_trace(go.Scatter(
        x=mean_recall.tolist(),
        y=mean_precision.tolist(),
        line=dict(color='rgb(44, 160, 44)', width=2),
        name=f'Average PR (AP = {mean_ap:.3f})',
        mode='lines',
    ))
    
    # Add confidence interval
    ci_x = np.concatenate([mean_recall, mean_recall[::-1]]).tolist()
    ci_y = np.concatenate([mean_precision + std_precision, (mean_precision - std_precision)[::-1]]).tolist()
    fig.add_trace(go.Scatter(
        x=ci_x,
        y=ci_y,
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
    
    # Generate curves with sample data
    create_average_roc_curve(sample_fold_histories)
    create_average_pr_curve(sample_fold_histories)
    print("Generated average ROC and PR curves in static directory.") 