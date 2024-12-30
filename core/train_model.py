import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Configure Django settings before importing any Django modules
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')

# Initialize Django
import django
django.setup()

# Import required modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm  # type: ignore
import torch.amp as amp  # Import amp directly
import torch.multiprocessing as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

# Import our model
from core.models import LungClassifierModel

# Set PyTorch to use all available CPU cores
torch.set_num_threads(16)  # i9 has good multi-threading

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True  # Enable auto-tuner
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
torch.backends.cudnn.allow_tf32 = True

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Set number of workers for data loading - using more cores for data loading
NUM_WORKERS = 8  # Increased from 4 to 8 for i9
BATCH_SIZE = 128  # Increased for RTX 4080 12GB
PIN_MEMORY = True  # Enable pinned memory for faster data transfer

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

def train_model(data_dir, epochs=30, batch_size=BATCH_SIZE):
    print(f"\nInitializing training with:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Workers: {NUM_WORKERS}")
    print(f"- Device: {device}")
    print(f"- Pin Memory: {PIN_MEMORY}")

    # Create output directories
    model_dir = os.path.join(project_root, 'models')
    static_dir = os.path.join(project_root, 'static')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # Enhanced data augmentation for training
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
        ], p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Validation transforms
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    try:
        # Create datasets
        train_dataset = XRayDataset(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        
        val_dataset = XRayDataset(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )

        print(f"\nDataset sizes:")
        print(f"- Training: {len(train_dataset)} images")
        print(f"- Validation: {len(val_dataset)} images")

        # Create data loaders with multiple workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=True,
            prefetch_factor=2  # Prefetch 2 batches per worker
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=True,
            prefetch_factor=2
        )

        # Initialize model, loss function, and optimizer
        model = LungClassifierModel().to(device)
        
        # Enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )

        # Initialize GradScaler for mixed precision training
        scaler = amp.GradScaler()

        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for inputs, labels in train_bar:
                try:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    
                    # Mixed precision training
                    with amp.autocast(device_type='cuda'):
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    scheduler.step()
                    
                    train_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    train_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.0 * train_correct / train_total:.2f}%',
                        'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                    })

                    # Clear GPU cache if memory is getting full
                    if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.95:
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"\nWARNING: GPU out of memory, skipping batch")
                        continue
                    else:
                        raise e
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for inputs, labels in val_bar:
                    try:
                        inputs, labels = inputs.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
                        
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
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(f"\nWARNING: GPU out of memory, skipping batch")
                            continue
                        else:
                            raise e
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100.0 * train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100.0 * val_correct / val_total
            
            # Save history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            history['lr'].append(scheduler.get_last_lr()[0])
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                if isinstance(model, nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(model_dir, 'lung_classifier_model.pth'))
            
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

            # Clear GPU cache between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        plt.subplot(1, 3, 3)
        plt.plot(history['lr'])
        plt.title('Learning rate')
        plt.ylabel('Learning rate')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, 'training_history.png'), dpi=300)
        
        return model, history

    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        data_dir = os.path.join(project_root, 'Chest_X-Ray_Images_(Pneumonia)_2_classes', 'chest_xray')
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory not found: {data_dir}")
            
        print(f"Training with data from: {data_dir}")
        model, history = train_model(data_dir)
        
        # Print final metrics
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)