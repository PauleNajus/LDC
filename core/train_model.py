import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

# Import required modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm  # type: ignore
import torch.cuda.amp as amp
import torch.multiprocessing as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import time
import json

# Add at the very top of the file before other imports
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable version check

# Define the model architecture (copied from models.py to make it independent)
class LungClassifierModel(nn.Module):
    def __init__(self):
        super(LungClassifierModel, self).__init__()
        
        # Enhanced initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Deeper residual blocks with more channels
        self.layer1 = self._make_layer(96, 96, 3)
        self.layer2 = self._make_layer(96, 192, 4, stride=2)
        self.layer3 = self._make_layer(192, 384, 6, stride=2)
        self.layer4 = self._make_layer(384, 768, 3, stride=2)
        
        # Enhanced classifier with dropout
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential stride
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x

# Optimize system resources
torch.set_num_threads(16)  # i9 has good multi-threading
mp.set_start_method('spawn', force=True)  # Better multiprocessing method

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Empty CUDA cache
torch.cuda.empty_cache()

def get_device_info():
    """Get and format device information for display."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = [
        "=" * 40,
        "DEVICE INFORMATION",
        "=" * 40,
        f"Device: {device}"
    ]
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info.extend([
            f"GPU: {gpu_name}",
            f"Memory Available: {memory_gb:.2f} GB",
            f"CUDA Version: {torch.version.cuda}",
            f"cuDNN Version: {torch.backends.cudnn.version()}",
        ])
    
    info.append("=" * 40)
    return device, "\n".join(info)

# Device configuration
device, device_info = get_device_info()

# Optimized parameters for RTX 4080 12GB
BATCH_SIZE = 192  # Reduced from 256 to improve stability
ACCUMULATION_STEPS = 3  # Increased for effective batch size of 576
NUM_WORKERS = 12  # Optimized for modern CPU
PIN_MEMORY = True
PREFETCH_FACTOR = 4
PATIENCE = 10  # Reduced since we're seeing good convergence by epoch 30
MIN_EPOCHS = 30  # Reduced based on observed convergence
MAX_EPOCHS = 100  # Reduced since we're unlikely to need 200 epochs
TARGET_ACCURACY = 0.96  # Updated to 96% target accuracy
CHECKPOINT_FREQ = 10  # Updated to save checkpoint every 10 epochs

# Memory optimization settings
torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM
torch.cuda.memory.set_per_process_memory_fraction(0.95)

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset directory not found: {root_dir}")
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            img, label = self.dataset[idx]
            img = np.array(img)
            
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            
            return img, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            raise e

def save_checkpoint(state, is_best, fold, filename='checkpoint.pth.tar'):
    """Save training checkpoint."""
    checkpoint_path = os.path.join(project_root, 'models', f'checkpoint_fold_{fold}.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(project_root, 'models', f'best_model_fold_{fold}.pth')
        torch.save(state['state_dict'], best_path)

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, scaler, epoch, fold, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    optimizer.zero_grad(set_to_none=True)

    # Learning rate warmup for first 5 epochs
    if epoch < 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = optimizer.param_groups[0]['lr'] * (epoch + 1) / 5

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Fold {fold+1} [Train]')
    for batch_idx, (inputs, labels) in enumerate(train_bar):
        # Move data to GPU asynchronously
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        
        # Mixed precision training
        with amp.autocast():
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss = loss / ACCUMULATION_STEPS  # Normalize loss for gradient accumulation
        
        # Scale loss and compute gradients
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)  # Unscale gradients for proper clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        train_loss += loss.item() * ACCUMULATION_STEPS
        with torch.no_grad():  # Ensure no memory is allocated for gradients during prediction
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Update progress bar
        train_bar.set_postfix({
            'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
            'acc': f'{100.0 * train_correct / train_total:.2f}%'
        })
        
        # Optional: clear cache periodically
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    return train_loss / len(train_loader), train_correct / train_total

def validate_one_fold(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # Ensure no memory is allocated for gradients during validation
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='[Validation]')
        for inputs, labels in val_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            with amp.autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * val_correct / val_total:.2f}%'
            })
        
        # Clear cache after validation
        torch.cuda.empty_cache()

    return val_loss / len(val_loader), val_correct / val_total

def save_training_metrics(model, history, fold_histories, model_dir, n_folds=5):
    """Save training metrics and visualizations."""
    static_dir = os.path.join(project_root, 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'training_history.png'))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    for fold_idx, fold_history in enumerate(fold_histories):
        fpr, tpr, _ = roc_curve(fold_history['true_labels'], fold_history['pred_probs'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Fold {fold_idx + 1} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    
    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    for fold_idx, fold_history in enumerate(fold_histories):
        precision, recall, _ = precision_recall_curve(fold_history['true_labels'], fold_history['pred_probs'])
        plt.plot(recall, precision, label=f'Fold {fold_idx + 1}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'model_evaluation.png'))
    plt.close()
    
    # Save confusion matrices
    plt.figure(figsize=(15, 5))
    for fold_idx, fold_history in enumerate(fold_histories):
        plt.subplot(1, n_folds, fold_idx + 1)
        cm = confusion_matrix(fold_history['true_labels'], fold_history['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'confusion_matrices.png'))
    plt.close()

def train_model(data_dir, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, n_folds=5):
    # Display device information once at the start
    print(device_info)
    
    print(f"\nInitializing training with:")
    print(f"- Max Epochs: {epochs}")
    print(f"- Batch size: {batch_size} (effective: {batch_size * ACCUMULATION_STEPS})")
    print(f"- Target Accuracy: {TARGET_ACCURACY * 100}%")
    print(f"- Early Stopping Patience: {PATIENCE}")
    print(f"- Number of folds: {n_folds}")
    print(f"- Checkpointing every {CHECKPOINT_FREQ} epochs")

    # Create output directories
    model_dir = os.path.join(project_root, 'models')
    static_dir = os.path.join(project_root, 'static')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # Enhanced data augmentation
    train_transform = A.Compose([
        A.Resize(height=320, width=320),  # Increased from 256x256
        A.RandomCrop(height=288, width=288),  # Increased from 224x224
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.1,
            scale=0.1,
            rotate=15,
            shear=0,  # Add shear if needed
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=320, width=320),  # Increased from 256x256
        A.CenterCrop(height=288, width=288),  # Increased from 224x224
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    try:
        # Create dataset
        full_dataset = XRayDataset(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        
        # Initialize K-Fold cross validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store fold results
        fold_results = []
        
        # Convert dataset to numpy array for KFold
        dataset_indices = np.arange(len(full_dataset))
        
        history = {
            'train_acc': [], 'val_acc': [], 
            'train_loss': [], 'val_loss': [],
            'learning_rates': [],
            'start_time': time.time()
        }
        
        fold_histories = []
        
        # Training loop for each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_indices)):
            print(f'\nFold {fold + 1}/{n_folds}')
            
            # Create data samplers for fold
            train_sampler = SubsetRandomSampler(train_idx.tolist())
            val_sampler = SubsetRandomSampler(val_idx.tolist())
            
            # Create data loaders for fold
            train_loader = DataLoader(
                full_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=True,
                prefetch_factor=PREFETCH_FACTOR
            )
            
            val_loader = DataLoader(
                full_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=True,
                prefetch_factor=PREFETCH_FACTOR
            )

            # Initialize model for fold
            model = LungClassifierModel().to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(device))
            optimizer = optim.AdamW(
                model.parameters(),
                lr=0.0005,
                weight_decay=0.02,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Initialize GradScaler for mixed precision training
            scaler = torch.cuda.amp.GradScaler()
            
            # Training history for fold
            fold_history = {
                'true_labels': [],
                'predictions': [],
                'pred_probs': []
            }

            best_val_acc = 0.0
            patience_counter = 0
            best_epoch = 0
            
            # Training loop for fold
            for epoch in range(epochs):
                # Train and validate
                train_loss, train_acc = train_one_fold(
                    model, train_loader, val_loader, criterion,
                    optimizer, scaler, epoch, fold, device
                )
                
                val_loss, val_acc = validate_one_fold(
                    model, val_loader, criterion, device
                )
                
                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                print(f'\nEpoch {epoch + 1}/{epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%')
                
                # Save checkpoint periodically
                if (epoch + 1) % CHECKPOINT_FREQ == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_val_acc': best_val_acc,
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'history': history,
                    }, False, fold)
                
                # Save best model and check early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_val_acc': best_val_acc,
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'history': history,
                    }, True, fold)
                else:
                    patience_counter += 1

                # Check if we've reached target accuracy
                if val_acc >= TARGET_ACCURACY and epoch >= MIN_EPOCHS:
                    print(f"\nReached target accuracy of {TARGET_ACCURACY * 100}% at epoch {epoch + 1}")
                    break

                # Early stopping check
                if patience_counter >= PATIENCE and epoch >= MIN_EPOCHS:
                    print(f"\nEarly stopping triggered. Best validation accuracy: {best_val_acc * 100:.2f}% at epoch {best_epoch + 1}")
                    break
            
            # Collect predictions and true labels for metrics
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    
                    fold_history['true_labels'].extend(labels.numpy())
                    fold_history['predictions'].extend(preds.cpu().numpy())
                    fold_history['pred_probs'].extend(probs.cpu().numpy())
            
            fold_histories.append(fold_history)
            
            # Track learning rates
            history['learning_rates'].extend([group['lr'] for group in optimizer.param_groups])
            
            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'best_val_acc': best_val_acc,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss
            })
            
            # Plot and save training history for fold
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title(f'Loss History - Fold {fold + 1}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Acc')
            plt.plot(history['val_acc'], label='Val Acc')
            plt.title(f'Accuracy History - Fold {fold + 1}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, f'training_history_fold_{fold + 1}.png'))
            plt.close()

        # Save and display cross-validation results
        results_df = pd.DataFrame(fold_results)
        print("\nCross-validation results:")
        print(results_df)
        print(f"\nMean validation accuracy: {results_df['best_val_acc'].mean() * 100:.2f}%")
        print(f"Std validation accuracy: {results_df['best_val_acc'].std() * 100:.2f}%")
        
        # Save final ensemble model (average of best models from each fold)
        ensemble_model = LungClassifierModel().to(device)
        if torch.cuda.device_count() > 1:
            ensemble_model = nn.DataParallel(ensemble_model)
        
        # Average the weights of best models from each fold
        state_dicts = []
        for fold in range(n_folds):
            state_dict = torch.load(
                os.path.join(model_dir, f'best_model_fold_{fold + 1}.pth'),
                map_location=device
            )
            state_dicts.append(state_dict)
        
        averaged_state_dict = {}
        for key in state_dicts[0].keys():
            averaged_state_dict[key] = sum(state_dict[key] for state_dict in state_dicts) / n_folds
        
        ensemble_model.load_state_dict(averaged_state_dict)
        torch.save(ensemble_model.state_dict(), os.path.join(model_dir, 'final_ensemble_model.pth'))

        # Save comprehensive metrics and visualizations
        save_training_metrics(ensemble_model, history, fold_histories, model_dir, n_folds)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

def main():
    print(device_info)
    
    # Set data directory
    data_dir = os.path.join(project_root, 'data', 'Chest_X-Ray_Images_(Pneumonia)_2_classes', 'chest_xray')
    print(f"Training with data from: {data_dir}")
    
    # Data augmentation and preprocessing
    train_transform = A.Compose([
        A.Resize(height=320, width=320),  # Increased from 256x256
        A.RandomCrop(height=288, width=288),  # Increased from 224x224
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.1,
            scale=0.1,
            rotate=15,
            shear=0,  # Add shear if needed
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=320, width=320),  # Increased from 256x256
        A.CenterCrop(height=288, width=288),  # Increased from 224x224
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Create datasets
    train_dataset = XRayDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = XRayDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    print("\nInitializing training with:")
    print(f"- Max Epochs: {MAX_EPOCHS}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Target Accuracy: {TARGET_ACCURACY*100}%")
    print(f"- Early Stopping Patience: {PATIENCE}")
    print(f"- Number of folds: 5")
    
    # Initialize model, optimizer, criterion
    model = LungClassifierModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Initialize mixed precision scaler
    scaler = amp.GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    try:
        for epoch in range(MAX_EPOCHS):
            train_loss, train_acc = train_one_fold(
                model, train_loader, val_loader, criterion,
                optimizer, scaler, epoch, 0, device
            )
            val_loss, val_acc = validate_one_fold(model, val_loader, criterion, device)
            
            print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Check early stopping conditions
            if epoch >= MIN_EPOCHS and (patience_counter >= PATIENCE or val_acc >= TARGET_ACCURACY):
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save training metrics
        metrics = {
            'best_val_accuracy': float(best_val_acc),
            'total_training_time': float(training_time),
            'gpu_memory_used_gb': float(torch.cuda.max_memory_allocated() / 1024**3),
            'convergence_epoch': epoch + 1
        }
        
        os.makedirs('static', exist_ok=True)
        with open('static/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return best_val_acc, training_time
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Train the lung classifier model')
        parser.add_argument('--data_dir', type=str, help='Path to the dataset directory')
        parser.add_argument('--epochs', type=int, default=MAX_EPOCHS, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
        parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
        args = parser.parse_args()
        
        # Use provided data_dir or fall back to default
        data_dir = args.data_dir if args.data_dir else os.path.join(project_root, 'data', 'Chest_X-Ray_Images_(Pneumonia)_2_classes', 'chest_xray')
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory not found: {data_dir}")
            
        print(f"Training with data from: {data_dir}")
        train_model(data_dir, epochs=args.epochs, batch_size=args.batch_size, n_folds=args.n_folds)
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)