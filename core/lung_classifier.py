from typing import Tuple, Union, Optional, BinaryIO, Any, cast
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import logging
import torch.cuda.amp as amp
import io

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
            model_logger.info(f'Loading model from {self.model_path}')
            self.model = torch.load(str(self.model_path), map_location=self.device)
            if isinstance(self.model, nn.Module):
                self.model.to(self.device)
                self.model.eval()
                model_logger.info('Model loaded successfully')
                model_logger.debug(f'Model architecture:\n{self.model}')
            else:
                raise TypeError("Loaded model is not an instance of nn.Module")
        except Exception as e:
            model_logger.error(f'Failed to load model: {str(e)}')
            raise

    def preprocess_image(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> torch.Tensor:
        try:
            pytorch_logger.debug('Starting image preprocessing')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                pytorch_logger.debug('Loaded image from path')
            elif isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert('RGB')
                pytorch_logger.debug('Loaded image from bytes')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pytorch_logger.debug('Converted numpy array to PIL Image')
            elif not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image type: {type(image)}")

            img_tensor = self.transform(image)
            if not isinstance(img_tensor, torch.Tensor):
                raise TypeError(f"Transformed image is not a tensor, got {type(img_tensor)}")
            pytorch_logger.debug(f'Image transformed to tensor of shape {img_tensor.shape}')
            img_tensor = img_tensor.unsqueeze(0)
            pytorch_logger.debug(f'Batch dimension added, final shape: {img_tensor.shape}')
            return img_tensor.to(self.device)
        except Exception as e:
            pytorch_logger.error(f'Image preprocessing failed: {str(e)}')
            raise

    @torch.no_grad()
    def predict(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[int, float]:
        try:
            if self.model is None:
                model_logger.error('Model not loaded')
                raise RuntimeError('Model not loaded')

            model_logger.info('Starting prediction')
            img_tensor = self.preprocess_image(image)
            
            with amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            model_logger.info(f'Prediction complete - Class: {predicted_class}, Confidence: {confidence:.4f}')
            pytorch_logger.debug(f'Raw outputs: {outputs.cpu().numpy()}')
            pytorch_logger.debug(f'Probabilities: {probabilities.cpu().numpy()}')
            
            return int(predicted_class), float(confidence)
        except Exception as e:
            model_logger.error(f'Prediction failed: {str(e)}')
            raise

    def __str__(self) -> str:
        return f'LungClassifier(device={self.device}, image_size={self.image_size})' 