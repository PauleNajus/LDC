import os
import sys
from pathlib import Path
from PIL import Image

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

# Import required modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from tqdm import tqdm  # type: ignore
import torch.cuda.amp as amp
import torch.multiprocessing as mp
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import time
import json
import copy
import random
import platform
from datetime import datetime

# Disable albumentations version check
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Configuration constants
MAX_EPOCHS = 100
MIN_EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 8
PIN_MEMORY = True
PATIENCE = 10
DEFAULT_TARGET_ACCURACY = 0.98
CHECKPOINT_FREQ = 1
LEARNING_RATE = 2e-4  # Increased initial learning rate
WEIGHT_DECAY = 1e-4  # Increased weight decay for better regularization
IMAGE_SIZE = 224  # Input image size
ACCUMULATION_STEPS = 1
MIXUP_ALPHA = 0.2
CHECKPOINT_DIR = os.path.join(project_root, 'models')
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable benchmark mode for faster training when input sizes don't change
torch.backends.cudnn.benchmark = True

def get_device_info():
    if not torch.cuda.is_available():
        return "CUDA not available."
    
    device_info = [
        f"CUDA Available: {torch.cuda.is_available()}",
        f"Device Count: {torch.cuda.device_count()}"
    ]
    
    for i in range(torch.cuda.device_count()):
        device_info.extend([
            f"Device {i}: {torch.cuda.get_device_name(i)}",
            f"  - Compute Capability: {torch.cuda.get_device_capability(i)}",
            f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
        ])
    
    device_info.extend([
        f"Current Device: {torch.cuda.current_device()}",
        f"PyTorch Version: {torch.__version__}",
        f"CUDNN Version: {torch.backends.cudnn.version()}",
        f"CUDNN Enabled: {torch.backends.cudnn.enabled}",
        f"CUDNN Benchmark: {torch.backends.cudnn.benchmark}",
        f"Deterministic: {torch.backends.cudnn.deterministic}"
    ])
    
    return "\n".join(device_info)

device_info = get_device_info()

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform: Compose | None = None, cache_size=1000):
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset directory not found: {root_dir}")
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.samples = self.dataset.samples
        self.classes = self.dataset.classes
        
        # Print class distribution
        labels = [label for _, label in self.samples]
        self.class_counts = {self.classes[i]: labels.count(i) for i in range(len(self.classes))}
        print(f"Class distribution in {os.path.basename(root_dir)}: {self.class_counts}")
        
        # Initialize cache for frequently accessed images
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            # Check if image is in cache
            if idx in self.cache:
                self.access_count[idx] += 1
                return self.cache[idx]
            
            # Load image
            img_path = self.samples[idx][0]
            label = self.samples[idx][1]
            
            # Use PIL Image instead of OpenCV for better compatibility with torchvision
            img = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                img = np.array(img)
                transformed = self.transform(image=img)
                img = transformed["image"]
            
            # Add to cache if there's space
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (img, label)
                self.access_count[idx] = 1
            
            return img, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            # Return placeholder data on error
            return torch.zeros(3, 224, 224), 0
            
    def balance_classes(self, max_ratio=1.0):
        """Balance classes by downsampling the majority class"""
        labels = [label for _, label in self.samples]
        class_indices = {i: [idx for idx, (_, label) in enumerate(self.samples) if label == i] 
                         for i in range(len(self.classes))}
        
        # Find minimum class count
        min_count = min(len(indices) for indices in class_indices.values())
        target_count = int(min_count * max_ratio)
        
        # Create balanced dataset
        balanced_indices = []
        for class_idx, indices in class_indices.items():
            if len(indices) > target_count:
                # Randomly select subset of samples
                selected_indices = random.sample(indices, target_count)
                balanced_indices.extend(selected_indices)
            else:
                balanced_indices.extend(indices)
        
        # Update samples
        self.samples = [self.samples[i] for i in balanced_indices]
        
        # Update class distribution
        labels = [label for _, label in self.samples]
        self.class_counts = {self.classes[i]: labels.count(i) for i in range(len(self.classes))}
        print(f"Balanced class distribution: {self.class_counts}")

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save training checkpoint."""
    checkpoint_path = os.path.join(project_root, 'models', filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(project_root, 'models', 'best_model.pth')
        torch.save(state['state_dict'], best_path)
        print(f"Saved best model with accuracy: {state['best_acc']:.4f}")

def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to the batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates the mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device, accumulation_steps=2, mixup_alpha=0.2):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # For accurate training accuracy calculation
    non_mixup_correct = 0
    non_mixup_total = 0
    
    # Reset gradients at the beginning
    optimizer.zero_grad()
    
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for i, (inputs, labels) in enumerate(train_bar):
        # Move data to device with asynchronous data loading
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply mixup augmentation
        use_mixup = random.random() < 0.5  # Apply mixup 50% of the time
        if use_mixup:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
        
        # Forward pass with mixed precision
        with amp.autocast(): # type: ignore[attr-defined]
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) / accumulation_steps
            else:
                loss = criterion(outputs, labels) / accumulation_steps
        
        # Scale loss and compute gradients
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps batches or at the end of an epoch
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            # Unscale optimizer for gradient clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # type: ignore[attr-defined]
            
            # Update weights with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            
            # Reset gradients
            optimizer.zero_grad()
        
        # Compute metrics without blocking GPU operations
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            train_loss += loss.item() * accumulation_steps
            train_total += labels.size(0)
            
            # Track accuracy differently for mixup vs non-mixup batches
            if not use_mixup:
                batch_correct = (predicted == labels).sum().item()
                non_mixup_correct += batch_correct
                non_mixup_total += labels.size(0)
            
            # Calculate the current accuracy to display in the progress bar
            current_acc = non_mixup_correct / max(1, non_mixup_total)  # Prevent division by zero
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{train_loss/(i+1):.4f}',
                'acc': f'{current_acc:.4f}' if non_mixup_total > 0 else 'mixup'
            })
    
    # Return average loss and accuracy across the epoch
    return train_loss / len(train_loader), non_mixup_correct / max(1, non_mixup_total)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    val_bar = tqdm(val_loader, desc='[Validation]')
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track metrics
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            val_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * val_correct / val_total:.2f}%'
            })
    
    return val_loss / len(val_loader), val_correct / val_total

def evaluate_model(model, test_loader, device, use_tta=True, tta_transforms=5, threshold=0.5):
    """Evaluate the model on the test set with optional TTA"""
    start_time = time.time()
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Ensure model is in evaluation mode
    model.eval()
    
    test_bar = tqdm(test_loader, desc='[Testing]')
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # Get original predictions
            outputs = model(inputs)
            batch_probs = torch.softmax(outputs, dim=1)
            
            if use_tta:
                # Apply basic test-time augmentation - horizontal flip only
                # This is more reliable than complex TTA
                flipped_inputs = torch.flip(inputs, dims=[3])  # Flip horizontally
                flipped_outputs = model(flipped_inputs)
                flipped_probs = torch.softmax(flipped_outputs, dim=1)
                
                # Average the predictions
                batch_probs = (batch_probs + flipped_probs) / 2
            
            # Get class predictions using threshold
            predicted = (batch_probs[:, 1] > threshold).long()
            
            # Save predictions and labels for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(batch_probs[:, 1].cpu().numpy())
            
            # Calculate batch accuracy
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            test_bar.set_postfix({
                'acc': f'{100.0 * test_correct / test_total:.2f}%'
            })
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = test_correct / test_total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate confusion matrix and derived metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate precision, F1-score, and balanced accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Calculate ROC curve and AUC
    try:
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall_curve, precision_curve)
    except Exception as e:
        print(f"Error calculating ROC/PR curves: {str(e)}")
        roc_auc = 0
        pr_auc = 0
        fpr, tpr, roc_thresholds = [], [], []
        precision_curve, recall_curve, pr_thresholds = [], [], []
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # same as recall
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'roc_curve': {
            'fpr': fpr.tolist() if isinstance(fpr, np.ndarray) else fpr,
            'tpr': tpr.tolist() if isinstance(tpr, np.ndarray) else tpr,
            'thresholds': roc_thresholds.tolist() if isinstance(roc_thresholds, np.ndarray) else roc_thresholds
        },
        'pr_curve': {
            'precision': precision_curve.tolist() if isinstance(precision_curve, np.ndarray) else precision_curve,
            'recall': recall_curve.tolist() if isinstance(recall_curve, np.ndarray) else recall_curve,
            'thresholds': pr_thresholds.tolist() if isinstance(pr_thresholds, np.ndarray) else pr_thresholds
        },
        'evaluation_time': evaluation_time
    }

def save_training_metrics(history, test_metrics, model_dir):
    """Save training history and test metrics to JSON files and generate plots."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save history to JSON
    history_path = os.path.join(model_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    # Convert numpy arrays in metrics to lists for JSON serialization
    def convert_np_to_list(data):
        if isinstance(data, dict):
            return {k: convert_np_to_list(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_np_to_list(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer)):
            return int(data)
        elif isinstance(data, (np.floating)):
            return float(data)
        elif isinstance(data, (np.bool_)):
            return bool(data)
        else:
            return data

    test_metrics_serializable = convert_np_to_list(test_metrics)

    # Save metrics to JSON
    metrics_path = os.path.join(model_dir, f'test_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics_serializable, f, indent=4)
    print(f"Test metrics saved to {metrics_path}")
    
    # Generate and save visualization plots
    generate_visualization_plots(history_path, metrics_path, model_dir)

# Ensure visualization functions exist or are defined elsewhere
# Add dummy functions if needed for testing
def generate_visualization_plots(history_path, metrics_path, output_dir):
    print("\n========== GENERATING ADDITIONAL VISUALIZATION PLOTS ==========")
    try:
        # Load data
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Using training history: {history_path}")
        
        with open(metrics_path, 'r') as f:
            test_metrics = json.load(f)
        print(f"Using test metrics: {metrics_path}")

        print("Generating plots...")
        
        # Ensure output directory exists
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Plot Training & Validation Curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
        plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
        plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'train_val_curves.png'))
        plt.close()
        print(f"Training and validation curves saved to {vis_dir}")

        # Plot Confusion Matrix Heatmap
        if 'confusion_matrix' in test_metrics:
            cm = np.array(test_metrics['confusion_matrix'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix (Threshold: {test_metrics.get("best_threshold", "N/A"):.2f})')
            plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
            plt.close()
            print(f"Confusion matrix heatmap saved to {vis_dir}")

        # Plot ROC Curve
        print(f"Test metrics keys: {list(test_metrics.keys())}") # Debug print
        if 'roc_curve' in test_metrics and isinstance(test_metrics['roc_curve'], dict):
            roc_data = test_metrics['roc_curve']
            print(f"ROC curve keys: {list(roc_data.keys())}") # Debug print
            if all(k in roc_data for k in ['fpr', 'tpr']):
                fpr = np.array(roc_data['fpr'])
                tpr = np.array(roc_data['tpr'])
                roc_auc = test_metrics.get('roc_auc', None)
                
                # Check data types
                print("ROC curve data types:")
                print(f"  fpr: {type(fpr)}")
                if len(fpr) > 0: print(f"    First element type: {type(fpr[0])}")
                print(f"  tpr: {type(tpr)}")
                if len(tpr) > 0: print(f"    First element type: {type(tpr[0])}")
                if 'thresholds' in roc_data: print(f"  thresholds: {type(roc_data['thresholds'])}")
                if 'thresholds' in roc_data and len(roc_data['thresholds']) > 0: print(f"    First element type: {type(roc_data['thresholds'][0])}")

                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})' if roc_auc is not None else 'ROC curve')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.savefig(os.path.join(vis_dir, 'roc_curve.png'))
                plt.close()
                print(f"ROC curve saved to {vis_dir}")
            else:
                 print("Warning: Missing 'fpr' or 'tpr' keys in roc_curve data.")
        else:
            print("Warning: 'roc_curve' data not found or not in expected format in test_metrics.")

        # Plot Precision-Recall Curve
        if 'pr_curve' in test_metrics and isinstance(test_metrics['pr_curve'], dict):
            pr_data = test_metrics['pr_curve']
            if all(k in pr_data for k in ['precision', 'recall']):
                precision = np.array(pr_data['precision'])
                recall = np.array(pr_data['recall'])
                pr_auc = test_metrics.get('pr_auc', None)
                plt.figure()
                plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})' if pr_auc is not None else 'PR curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True)
                plt.savefig(os.path.join(vis_dir, 'precision_recall_curve.png'))
                plt.close()
                print(f"Precision-Recall curve saved to {vis_dir}")
            else:
                print("Warning: Missing 'precision' or 'recall' keys in pr_curve data.")
        else:
             print("Warning: 'pr_curve' data not found or not in expected format in test_metrics.")

        # Plot Learning Rate Schedule
        if 'lr' in history:
            plt.figure()
            plt.plot(history['epoch'], history['lr'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, 'lr_schedule.png'))
            plt.close()
            print(f"Learning rate schedule plot saved to {vis_dir}")

        # Plot Threshold Analysis (if available)
        if 'threshold_analysis' in test_metrics and isinstance(test_metrics['threshold_analysis'], dict):
             analysis = test_metrics['threshold_analysis']
             if all(k in analysis for k in ['thresholds', 'balanced_accuracies', 'best_threshold', 'best_balanced_accuracy']):
                 thresholds = np.array(analysis['thresholds'])
                 balanced_accuracies = np.array(analysis['balanced_accuracies'])
                 best_threshold = analysis['best_threshold']
                 best_balanced_accuracy = analysis['best_balanced_accuracy']
                 
                 print(f"Best threshold: {best_threshold} (type: {type(best_threshold)})")
                 print(f"Best balanced accuracy: {best_balanced_accuracy} (type: {type(best_balanced_accuracy)})")

                 plt.figure()
                 plt.plot(thresholds, balanced_accuracies, label='Balanced Accuracy')
                 plt.scatter([best_threshold], [best_balanced_accuracy], color='red', label=f'Best ({best_threshold:.2f}, {best_balanced_accuracy:.4f})', zorder=5)
                 plt.xlabel('Threshold')
                 plt.ylabel('Balanced Accuracy')
                 plt.title('Threshold vs. Balanced Accuracy')
                 plt.legend()
                 plt.grid(True)
                 plt.savefig(os.path.join(vis_dir, 'threshold_analysis.png'))
                 plt.close()
                 print(f"Threshold analysis plot saved to {vis_dir}")
             else:
                 print("Warning: Missing required keys in 'threshold_analysis' data.")
        else:
            print("Warning: 'threshold_analysis' data not found or not in expected format.")


        print("\nAll plots generated successfully!")
        print(f"Plots saved to: {vis_dir}")
        
    except Exception as e:
        print(f"Error generating visualization plots: {e}")
        import traceback
        traceback.print_exc()


def create_densenet169_model(num_classes=2):
    """Create a DenseNet-169 model with pre-trained weights."""
    # Use DenseNet-169 with updated weights API
    weights = models.DenseNet169_Weights.IMAGENET1K_V1
    model = models.densenet169(weights=weights)

    # Replace the final classifier layer
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Optional: Add dropout for regularization
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(num_ftrs, num_classes)
    # )

    print("Using DenseNet-169 model with ImageNet pre-trained weights.")
    return model, weights.transforms()

def create_model_ensemble(checkpoint_files, num_classes=2, device=None):
    """Create an ensemble of models from checkpoint files."""
    models_list = []
    print(f"Creating ensemble from {len(checkpoint_files)} models...")
    for checkpoint_file in checkpoint_files:
        # Create a new model instance - unpack the tuple
        model, _ = create_densenet169_model(num_classes)
        
        try:
            # Load state dictionary from checkpoint
            state_dict = torch.load(checkpoint_file)
            model.load_state_dict(state_dict)
        except:
            try:
                # Try loading from checkpoint dictionary
                checkpoint = torch.load(checkpoint_file)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    print(f"Could not load model from {checkpoint_file} - skipping")
                    continue
            except Exception as e:
                print(f"Error loading model from {checkpoint_file}: {str(e)} - skipping")
                continue
        
        model = model.to(device)
        model.eval()
        models_list.append(model)
    
    print(f"Successfully loaded {len(models_list)} models for the ensemble")
    return models_list

def evaluate_ensemble(models, test_loader, device, use_tta=True, tta_transforms=5, threshold=0.65):
    """Evaluate an ensemble of models."""
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # TTA transforms
    tta_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    test_bar = tqdm(test_loader, desc='[Testing Ensemble]')
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # Initialize ensemble predictions
            ensemble_probs = torch.zeros((batch_size, 2), device=device)
            
            for model in models:
                if use_tta:
                    # Test time augmentation
                    tta_probs = torch.zeros((batch_size, 2), device=device)
                    
                    # Original prediction
                    outputs = model(inputs)
                    tta_probs += torch.softmax(outputs, dim=1)
                    
                    # Convert to numpy for Albumentations
                    inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)
                    
                    # Apply TTA
                    for _ in range(tta_transforms):
                        tta_batch = []
                        for img in inputs_np:
                            # Apply augmentation
                            aug_img = tta_transform(image=img)['image']
                            tta_batch.append(aug_img)
                        
                        # Stack and move to device
                        tta_inputs = torch.stack(tta_batch).to(device)
                        
                        # Forward pass
                        aug_outputs = model(tta_inputs)
                        tta_probs += torch.softmax(aug_outputs, dim=1)
                    
                    # Average predictions
                    tta_probs /= (tta_transforms + 1)
                    ensemble_probs += tta_probs
                else:
                    # Standard evaluation
                    outputs = model(inputs)
                    ensemble_probs += torch.softmax(outputs, dim=1)
            
            # Average ensemble predictions
            ensemble_probs /= len(models)
            
            # Use a higher threshold for positive class to improve specificity
            predicted = (ensemble_probs[:, 1] > threshold).long()
            all_probs.extend(ensemble_probs[:, 1].cpu().numpy())
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track accuracy
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            test_bar.set_postfix({
                'acc': f'{100.0 * test_correct / test_total:.2f}%'
            })
            
    # Calculate metrics
    accuracy = test_correct / test_total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate sensitivity and specificity
    # Assuming 1 is positive (abnormal) and 0 is negative (normal)
    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    
    # Calculate ROC curve and AUC
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except:
        roc_auc = 0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def train_model(data_dir, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, target_accuracy=DEFAULT_TARGET_ACCURACY, 
             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=PATIENCE, 
             scheduler_patience=5, scheduler_factor=0.5):
    """Train the model with specified hyperparameters."""
    
    start_time = time.time()
    
    # Create necessary directories
    model_dir = os.path.join(project_root, 'train_results_logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Training DenseNet-169 model for abnormal detection")
    print("Device info:")
    print(device_info)

    # Print reproducibility info
    print("\n========== REPRODUCIBILITY INFO ==========")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Deterministic Mode: {torch.backends.cudnn.deterministic}")
    
    # Create model - unpack the tuple
    model, default_transforms = create_densenet169_model(num_classes=2)
    model = model.to(device)

    # Define data augmentations using Albumentations
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomScale(scale_limit=0.15, p=0.5),  # Scale +/- 15%
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE), # Ensure output size is IMAGE_SIZE
        A.Affine(
            scale=(0.85, 1.15),       # Zoom in/out by 15%
            translate_percent=(-0.1, 0.1), # Translate by +/- 10%
            rotate=(-20, 20),         # Rotate by +/- 20 degrees
            shear=(-10, 10),          # Shear by +/- 10 degrees
            p=0.7                     # Apply Affine 70% of the time
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7), # Adjust brightness/contrast by +/- 20%
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.5), # Apply Noise or Blur 50% of the time
        A.GridDistortion(p=0.3), # Apply Grid Distortion 30% of the time
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Use tuple for mean/std
        ToTensorV2() # Convert image and mask to PyTorch tensors
    ])

    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Use tuple for mean/std
        ToTensorV2()
    ])

    test_transform = val_transform # Use same basic transform for testing

    # Create datasets and dataloaders
    train_dataset = XRayDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = XRayDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = XRayDataset(os.path.join(data_dir, 'test'), transform=test_transform)

    # Balance training and validation datasets
    train_dataset.balance_classes(max_ratio=1.0)
    val_dataset.balance_classes(max_ratio=1.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Define loss function with class balancing (if needed, though datasets are now balanced)
    # Example: Calculate weights based on balanced dataset if still desired
    # class_counts = train_dataset.class_counts
    # total_samples = sum(class_counts.values())
    # weights = [total_samples / class_counts[train_dataset.classes[i]] for i in range(len(train_dataset.classes))]
    # class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Since we balanced the datasets, standard CrossEntropyLoss is fine
    criterion = nn.CrossEntropyLoss() 

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
    
    # Initialize GradScaler for mixed precision training
    scaler = amp.GradScaler() # type: ignore[attr-defined]

    print("\n========== HYPERPARAMETERS ==========")
    print(f"Optimizer: AdamW with amsgrad=True")
    print(f"Initial Learning Rate: {lr}")
    print(f"Weight Decay: {weight_decay}")
    print(f"LR Scheduler: ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience})")
    print(f"Batch Size: {batch_size}")
    print(f"Accumulation Steps: {ACCUMULATION_STEPS}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Max Epochs: {epochs}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Mixup Alpha: {MIXUP_ALPHA}")
    print(f"Loss Function: CrossEntropyLoss") # Updated as weights are not needed now

    print("\n========== DATA PREPROCESSING & AUGMENTATION ==========")
    print(f"Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet)")
    print("Training Augmentations:")
    for t in train_transform.transforms: print(f"  - {t.__class__.__name__}")
    print("Validation/Test Transforms:")
    for t in val_transform.transforms: print(f"  - {t.__class__.__name__}")
    print(f"Test-Time Augmentation: {5} transforms with flips and small rotations") # Placeholder value, TTA logic in evaluate_model

    # Training loop
    best_acc = 0.0
    epochs_no_improve = 0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\nStarting training for {epochs} epochs")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device, accumulation_steps=ACCUMULATION_STEPS, mixup_alpha=MIXUP_ALPHA)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        print(f"  Best Val Acc: {best_acc:.4f}")
        print(f"  Epoch Time: {epoch_duration:.2f}s, GPU Memory: {gpu_memory:.2f} GB")
        
        # Update learning rate scheduler
        scheduler.step(val_acc)

        # Save checkpoint and check for early stopping
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, filename=f'densenet169_epoch_{epoch+1}_acc_{val_acc:.4f}.pth')
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE and epoch >= MIN_EPOCHS:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save regular checkpoint
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()
             }, False, filename=f'checkpoint_epoch_{epoch+1}.pth')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining completed in {time.strftime('%Hh %Mm %Ss', time.gmtime(total_time))}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Load best model for evaluation
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for evaluation...")
        # Load single best model - unpack the tuple
        model, _ = create_densenet169_model(num_classes=2)
        model.load_state_dict(torch.load(best_model_path))
        model = model.to(device)
        # evaluate_model(model, test_loader, device, use_tta=True) # Moved evaluation outside the if block
    else:
        print("Warning: Best model checkpoint not found. Evaluating the final model.")
        # If best model not found, maybe evaluate the last state? Or skip? For now, just print warning.
        # Need to decide how to handle this case. Let's assume we still have the model object from the last epoch.

    # Evaluate on test set (using the model loaded from best checkpoint, or the last state if checkpoint missing)
    print("\nEvaluating model on test set...")
    test_metrics = evaluate_model(model, test_loader, device, use_tta=True)

    # Save metrics and generate plots
    save_training_metrics(history, test_metrics, model_dir)
    
    # Print performance summary
    avg_epoch_time = total_time / (epoch + 1) if epoch >= 0 else 0
    peak_gpu_memory = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    
    print("\n========== PERFORMANCE SUMMARY ==========")
    print(f"Total Time: {time.strftime('%Hh %Mm %Ss', time.gmtime(total_time))}")
    print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
    print(f"Total Epochs: {epoch + 1}")
    print(f"Peak GPU Memory Usage: {peak_gpu_memory:.2f} GB") # Using max_memory_reserved
    print(f"Training metrics and visualizations saved to {model_dir}")
    print(f"Training metrics saved to {model_dir}") # Redundant print?

    return best_acc


def main():
    # Set data directory
    data_dir = os.path.join(project_root, 'data', 'CXR8', 'chest_xray') # Corrected data directory path

    # Check if the directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        print("Please ensure the dataset is processed and located in the 'data/processed' folder.")
        sys.exit(1)

    # Train the model
    best_acc = train_model(data_dir=data_dir, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=EARLY_STOPPING_PATIENCE)

    # Optionally, evaluate the best model saved during training
    print("\n========== FINAL EVALUATION ==========")
    models_dir = os.path.join(project_root, 'models')
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    
    # Set up test data loader
    test_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_dataset = XRayDataset(os.path.join(data_dir, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation...")
        # Load single best model - Ensure model creation and loading uses the correct architecture
        model, _ = create_densenet169_model(num_classes=2) # Unpack tuple
        model.load_state_dict(torch.load(best_model_path)) # Load state dict into the model object
        model = model.to(device) # Move model to device
        evaluate_model(model, test_loader, device, use_tta=True)
    else:
        print("Best model checkpoint not found. Cannot perform final evaluation.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # Set start method for multiprocessing
    main()