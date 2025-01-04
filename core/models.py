from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.contrib.auth import get_user_model
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time

def get_default_user():
    User = get_user_model()
    return User.objects.first()

class User(AbstractUser):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.EmailField(unique=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.username

class XRayImage(models.Model):
    image = models.ImageField(upload_to='xray_images/')
    patient_name = models.CharField(max_length=100, default="No data")
    patient_surname = models.CharField(max_length=100, default="No data")
    patient_id = models.CharField(max_length=50, default="No data")
    patient_date_of_birth = models.CharField(max_length=20, default="No data")
    patient_gender = models.CharField(max_length=20, default="No data")
    xray_date = models.CharField(max_length=20, default="No data")
    prediction = models.CharField(max_length=20, blank=True, default="No data")
    confidence = models.FloatField(default=0.0)
    normal_probability = models.FloatField(default=0.0)
    pneumonia_probability = models.FloatField(default=0.0)
    processing_time = models.FloatField(default=0.0)  # in seconds
    image_size = models.CharField(max_length=50, default="No data")  # store as "WxH"
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"XRay Image - {self.uploaded_at}"

    def get_extension(self):
        """Get the file extension without the dot."""
        name = self.image.name
        return name.split('.')[-1] if '.' in name else ''

    def get_file_size_mb(self):
        """Get the file size in MB with 2 decimal places."""
        try:
            size_bytes = self.image.size
            size_mb = size_bytes / (1024 * 1024)  # Convert to MB
            return f"{size_mb:.2f}"
        except:
            return "0.00"

class LungClassifierModel(nn.Module):
    def __init__(self):
        super(LungClassifierModel, self).__init__()
        
        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
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

class LungClassifier:
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def build_model(self):
        model = LungClassifierModel()
        model = model.to(self.device)
        self.model = model
        return model

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)

    def predict(self, image_path):
        # Build or load model if not already done
        if self.model is None:
            self.build_model()
            if os.path.exists(settings.MODEL_PATH):
                state_dict = torch.load(settings.MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                raise Exception("Model weights not found!")

        # Ensure model exists before proceeding
        if self.model is None:
            raise Exception("Failed to initialize model!")

        # Get image size before preprocessing
        with Image.open(image_path) as img:
            original_size = f"{img.width}x{img.height}"

        # Start timing
        start_time = time.time()
        
        self.model.eval()
        with torch.no_grad():
            preprocessed_image = self.preprocess_image(image_path)
            logits = self.model(preprocessed_image)[0][0]
            prediction = torch.sigmoid(logits).item()  # Apply sigmoid to get probability
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Convert prediction to class label and confidence
        pneumonia_prob = prediction
        normal_prob = 1 - prediction
        class_idx = 1 if pneumonia_prob > 0.5 else 0
        confidence = pneumonia_prob if pneumonia_prob > 0.5 else normal_prob
        
        return {
            'class': self.classes[class_idx],
            'confidence': float(confidence * 100),
            'normal_probability': float(normal_prob * 100),
            'pneumonia_probability': float(pneumonia_prob * 100),
            'processing_time': float(processing_time),
            'image_size': original_size
        }