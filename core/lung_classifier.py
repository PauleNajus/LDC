from typing import Tuple, Union, Optional, BinaryIO, Any, cast
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
from PIL import Image
import logging
import torch.cuda.amp as amp
import io
import time

# Set up loggers
pytorch_logger = logging.getLogger('pytorch')
model_logger = logging.getLogger('model')

class LungClassifier:
    def __init__(self, model_path: Union[str, Path], image_size: Tuple[int, int] = (224, 224)):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pytorch_logger.info(f'Using device: {self.device}')
        if torch.cuda.is_available():
            pytorch_logger.info(f'GPU: {torch.cuda.get_device_name()}')
            pytorch_logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
            pytorch_logger.info(f'CUDA Version: {torch.version.cuda}')
            pytorch_logger.info(f'Current GPU Memory Usage:')
            pytorch_logger.info(f' - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
            pytorch_logger.info(f' - Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')
        else:
            pytorch_logger.warning('CUDA is not available. Running on CPU may be slower.')
        
        pytorch_logger.info(f'PyTorch Version: {torch.__version__}')
        
        self.model: Optional[nn.Module] = None
        self.scaler = amp.GradScaler()
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pytorch_logger.info('Initialized transform pipeline')
        self.load_model()

    def load_model(self) -> None:
        try:
            model_logger.info('\n=== Loading Model ===')
            model_logger.info(f'Model Path: {self.model_path}')
            model_logger.info(f'File Size: {self.model_path.stat().st_size / 1024 / 1024:.2f} MB')
            
            start_time = time.time()
            
            # Initialize DenseNet121 model with the correct number of classes (2 for binary classification)
            base_model = models.densenet121(weights=None)
            num_features = base_model.classifier.in_features
            
            # Replace classifier for binary classification (pneumonia vs normal)
            base_model.classifier = nn.Sequential(  # type: ignore
                nn.Dropout(p=0.2),
                nn.Linear(num_features, 2)  # 2 classes: normal and pneumonia
            )
            
            # Load the state dictionary
            state_dict = torch.load(str(self.model_path), map_location=self.device)
            
            # Check if the loaded file is a state_dict or a full model
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                # Handle checkpoint format where model weights are in 'state_dict' key
                state_dict = state_dict['state_dict']
            
            # Apply the weights to our model
            base_model.load_state_dict(state_dict)
            self.model = base_model
            
            load_time = time.time() - start_time
            
            self.model.to(self.device)
            self.model.eval()
            
            # Count model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            model_logger.info('\n=== Model Details ===')
            model_logger.info(f'Model Type: {type(self.model).__name__}')
            model_logger.info(f'Total Parameters: {total_params:,}')
            model_logger.info(f'Trainable Parameters: {trainable_params:,}')
            model_logger.info(f'Model Load Time: {load_time:.3f} seconds')
            model_logger.info(f'Model Architecture:\n{self.model}')
            
            if torch.cuda.is_available():
                model_logger.info('\n=== GPU Memory After Model Load ===')
                model_logger.info(f'Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')
                model_logger.info(f'Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB')
            
            model_logger.info('Model loaded successfully')
                
        except Exception as e:
            model_logger.error(f'Failed to load model: {str(e)}')
            model_logger.error(f'Error type: {type(e).__name__}')
            model_logger.error(f'Error details: {str(e)}')
            raise

    def preprocess_image(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> torch.Tensor:
        try:
            pytorch_logger.info('\n=== Starting Image Preprocessing ===')
            start_time = time.time()
            
            # Log input details
            original_type = type(image).__name__
            pytorch_logger.info(f'Input Type: {original_type}')
            
            # Convert to PIL Image first
            if isinstance(image, str):
                pytorch_logger.info(f'Input Path: {image}')
                pytorch_logger.info(f'File Size: {Path(image).stat().st_size / 1024:.2f} KB')
                image = Image.open(image).convert('RGB')
            elif isinstance(image, bytes):
                pytorch_logger.info(f'Input Bytes Size: {len(image) / 1024:.2f} KB')
                image = Image.open(io.BytesIO(image)).convert('RGB')
            elif isinstance(image, np.ndarray):
                pytorch_logger.info(f'Input Array Shape: {image.shape}')
                pytorch_logger.info(f'Array Type: {image.dtype}')
                pytorch_logger.info(f'Value Range: [{image.min()}, {image.max()}]')
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, Image.Image):
                pytorch_logger.info('Input is already a PIL Image')
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")

            # Now image is guaranteed to be a PIL Image
            pytorch_logger.info('\n=== Image Properties ===')
            if isinstance(image, Image.Image):  # Type check to satisfy linter
                pytorch_logger.info(f'Mode: {image.mode}')
                pytorch_logger.info(f'Size: {image.size}')
            pytorch_logger.info(f'Target Size: {self.image_size}')
            
            # Transform the image
            transform_start = time.time()
            img_tensor = self.transform(image)
            transform_time = time.time() - transform_start
            
            if not isinstance(img_tensor, torch.Tensor):
                raise TypeError(f"Transformed image is not a tensor, got {type(img_tensor)}")
            
            pytorch_logger.info('\n=== Tensor Properties ===')
            pytorch_logger.info(f'Shape: {img_tensor.shape}')
            pytorch_logger.info(f'Dtype: {img_tensor.dtype}')
            pytorch_logger.info(f'Device: {img_tensor.device}')
            pytorch_logger.info(f'Statistics:')
            pytorch_logger.info(f' - Min: {img_tensor.min():.3f}')
            pytorch_logger.info(f' - Max: {img_tensor.max():.3f}')
            pytorch_logger.info(f' - Mean: {img_tensor.mean():.3f}')
            pytorch_logger.info(f' - Std: {img_tensor.std():.3f}')
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            pytorch_logger.info(f'Final Shape: {img_tensor.shape}')
            
            # Log timing
            total_time = time.time() - start_time
            pytorch_logger.info('\n=== Preprocessing Timing ===')
            pytorch_logger.info(f'Transform Time: {transform_time:.3f} seconds')
            pytorch_logger.info(f'Total Time: {total_time:.3f} seconds')
            pytorch_logger.info('=== Preprocessing Complete ===\n')
            
            return img_tensor.to(self.device)
        except Exception as e:
            pytorch_logger.error('\n=== Preprocessing Error ===')
            pytorch_logger.error(f'Error Type: {type(e).__name__}')
            pytorch_logger.error(f'Error Message: {str(e)}')
            pytorch_logger.error(f'Input Type: {original_type}')
            raise

    @torch.no_grad()
    def predict(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[int, float]:
        try:
            if self.model is None:
                model_logger.error('Model not loaded')
                raise RuntimeError('Model not loaded')

            model_logger.info('\n=== Starting Prediction Process ===')
            total_start = time.time()
            
            # Log initial GPU status and memory
            initial_memory = 0
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                model_logger.info('\n=== Initial GPU Status ===')
                model_logger.info(f'Memory Allocated: {initial_memory / 1024**2:.1f} MB')
                model_logger.info(f'Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB')
                model_logger.info(f'Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB')
                try:
                    model_logger.info(f'GPU Utilization: {torch.cuda.utilization()}%')
                except ModuleNotFoundError:
                    model_logger.warning('Could not get GPU utilization: pynvml module not found')
                except Exception as e:
                    model_logger.warning(f'Could not get GPU utilization: {str(e)}')
            
            # Preprocess image
            preprocess_start = time.time()
            img_tensor = self.preprocess_image(image)
            preprocess_time = time.time() - preprocess_start
            
            # Run inference
            with amp.autocast(enabled=torch.cuda.is_available()):
                model_logger.info('\n=== Running Inference ===')
                inference_start = time.time()
                outputs = self.model(img_tensor)
                inference_time = time.time() - inference_start
                
                model_logger.info('\n=== Raw Model Output ===')
                model_logger.info(f'Shape: {outputs.shape}')
                model_logger.info(f'Type: {outputs.dtype}')
                model_logger.info(f'Range: [{outputs.min():.3f}, {outputs.max():.3f}]')
                
                # Calculate probabilities
                softmax_start = time.time()
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                softmax_time = time.time() - softmax_start
            
            # Log final GPU status
            if torch.cuda.is_available():
                model_logger.info('\n=== Final GPU Status ===')
                model_logger.info(f'Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')
                model_logger.info(f'Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB')
                model_logger.info(f'Memory Change: {(torch.cuda.memory_allocated() - initial_memory) / 1024**2:+.1f} MB')
            
            # Log prediction results
            model_logger.info('\n=== Prediction Results ===')
            model_logger.info(f'Predicted Class: {predicted_class}')
            model_logger.info(f'Confidence: {confidence:.4f}')
            model_logger.info('Class Probabilities:')
            for i, prob in enumerate(probabilities[0].cpu().numpy()):
                model_logger.info(f' - Class {i}: {prob:.4f}')
            
            # Log timing breakdown
            total_time = time.time() - total_start
            model_logger.info('\n=== Timing Breakdown ===')
            model_logger.info(f'Preprocessing Time: {preprocess_time:.3f} seconds')
            model_logger.info(f'Inference Time: {inference_time:.3f} seconds')
            model_logger.info(f'Softmax Time: {softmax_time:.3f} seconds')
            model_logger.info(f'Total Time: {total_time:.3f} seconds')
            model_logger.info('=== Prediction Complete ===\n')
            
            return int(predicted_class), float(confidence)
        except Exception as e:
            model_logger.error('\n=== Prediction Error ===')
            model_logger.error(f'Error Type: {type(e).__name__}')
            model_logger.error(f'Error Message: {str(e)}')
            if torch.cuda.is_available():
                model_logger.error(f'GPU Memory State: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated')
            raise

    def __str__(self) -> str:
        return f'LungClassifier(device={self.device}, image_size={self.image_size})' 