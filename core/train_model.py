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
import torch.amp as amp
import torch.multiprocessing as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

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
BATCH_SIZE = 128  # Optimized for RTX 4080 12GB with larger model
ACCUMULATION_STEPS = 1  # No need for accumulation with optimized batch size
NUM_WORKERS = 12  # Optimized for modern CPU
PIN_MEMORY = True
PREFETCH_FACTOR = 4
PATIENCE = 15  # Early stopping patience
MIN_EPOCHS = 50  # Minimum number of epochs before early stopping
MAX_EPOCHS = 200  # Maximum number of epochs
TARGET_ACCURACY = 0.98  # Target accuracy threshold

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

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, epoch, fold, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    optimizer.zero_grad(set_to_none=True)

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Fold {fold+1} [Train]')
    for batch_idx, (inputs, labels) in enumerate(train_bar):
        inputs, labels = inputs.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
        
        # Mixed precision training
        with amp.autocast(device_type='cuda'):
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss = loss / ACCUMULATION_STEPS  # Normalize loss for gradient accumulation
        
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        train_loss += loss.item() * ACCUMULATION_STEPS
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        train_bar.set_postfix({
            'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
            'acc': f'{100.0 * train_correct / train_total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    return train_loss / len(train_loader), train_correct / train_total

def validate_one_fold(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='[Validation]')
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
            
            with amp.autocast(device_type='cuda'):
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

    return val_loss / len(val_loader), val_correct / val_total

def train_model(data_dir, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, n_folds=5):
    # Display device information once at the start
    print(device_info)
    
    print(f"\nInitializing training with:")
    print(f"- Max Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Target Accuracy: {TARGET_ACCURACY * 100}%")
    print(f"- Early Stopping Patience: {PATIENCE}")
    print(f"- Number of folds: {n_folds}")

    # Create output directories
    model_dir = os.path.join(project_root, 'models')
    static_dir = os.path.join(project_root, 'static')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # Enhanced data augmentation
    train_transform = A.Compose([
        A.Resize(224, 224),  # Simpler resize instead of RandomResizedCrop
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
            A.GaussNoise(p=1),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            A.ElasticTransform(p=1)
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
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
            
            # Enhanced learning rate scheduler
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.002,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.2,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )

            # Initialize GradScaler for mixed precision training
            scaler = torch.cuda.amp.GradScaler()
            
            # Training history for fold
            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [],
                'lr': []
            }

            best_val_acc = 0.0
            patience_counter = 0
            best_epoch = 0
            
            # Training loop for fold
            for epoch in range(epochs):
                # Train and validate
                train_loss, train_acc = train_one_fold(
                    model, train_loader, val_loader, criterion,
                    optimizer, scheduler, scaler, epoch, fold, device
                )
                
                val_loss, val_acc = validate_one_fold(
                    model, val_loader, criterion, device
                )
                
                # Update history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['lr'].append(scheduler.get_last_lr()[0])
                
                print(f'\nEpoch {epoch + 1}/{epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%')
                
                # Save best model and check early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_dir, f'best_model_fold_{fold + 1}.pth')
                    )
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