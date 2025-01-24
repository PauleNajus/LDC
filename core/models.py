from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.contrib.auth import get_user_model
from django.core.validators import MinLengthValidator, RegexValidator
from django.core.exceptions import ValidationError
from typing import Optional
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import logging
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.db.models import AutoField, DateTimeField, JSONField, IntegerField
from datetime import datetime

logger = logging.getLogger('core')

def get_default_user():
    """Get the ID of the first user or None if no users exist."""
    User = get_user_model()
    try:
        first_user = User.objects.first()
        return first_user.pk if first_user else None
    except Exception as e:
        logger.error(f"Error getting default user: {str(e)}")
        return None

def validate_image_extension(value):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    ext = os.path.splitext(value.name)[1].lower()
    if ext not in valid_extensions:
        raise ValidationError(f'Unsupported file extension. Allowed extensions are: {", ".join(valid_extensions)}')

def validate_image_size(value):
    if value.size > settings.FILE_UPLOAD_MAX_MEMORY_SIZE:
        max_size_mb = settings.FILE_UPLOAD_MAX_MEMORY_SIZE / (1024 * 1024)
        raise ValidationError(f'File size cannot exceed {max_size_mb}MB')

class User(AbstractUser):
    """Custom user model with additional fields and functionality."""
    last_password_change: models.DateTimeField
    password_history: models.JSONField
    failed_login_attempts: models.IntegerField
    last_failed_login: Optional[models.DateTimeField]
    security_questions: models.JSONField
    
    first_name = models.CharField(max_length=30, validators=[MinLengthValidator(2)])
    last_name = models.CharField(max_length=30, validators=[MinLengthValidator(2)])
    email = models.EmailField(unique=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    last_password_change = models.DateTimeField(default=timezone.now)
    password_history = models.JSONField(default=list)  # Store hashed passwords
    failed_login_attempts = models.IntegerField(default=0)
    last_failed_login = models.DateTimeField(null=True, blank=True)
    security_questions = models.JSONField(default=dict)  # Store security Q&A
    language_preference = models.CharField(
        _('language preference'),
        max_length=10,
        choices=settings.LANGUAGES,
        default=settings.LANGUAGE_CODE,
        help_text=_('Preferred language for the user interface.')
    )

    class Meta:
        verbose_name = _('user')
        verbose_name_plural = _('users')
        ordering = ['-date_joined']
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['username']),
            models.Index(fields=['date_joined']),
            models.Index(fields=['last_password_change']),
        ]

    def __str__(self):
        return f"{self.get_full_name()} ({self.username})"

    def get_full_name(self):
        return f"{self.first_name} {self.last_name}".strip()

    def save(self, *args, **kwargs):
        self.first_name = self.first_name.title()
        self.last_name = self.last_name.title()
        self.email = self.email.lower()
        super().save(*args, **kwargs)
    
    def set_password(self, raw_password):
        """Override set_password to track password history."""
        if self.password:
            # Store the current password in history
            history = self.password_history
            history.append({
                'password': self.password,
                'date': timezone.now().isoformat()
            })
            # Keep only the last 5 passwords
            self.password_history = history[-5:]
        
        # Update last password change date
        self.last_password_change = timezone.now()
        
        super().set_password(raw_password)
    
    def check_password_history(self, raw_password):
        """Check if a password has been used before."""
        from django.contrib.auth.hashers import check_password
        
        for item in self.password_history:
            if check_password(raw_password, item['password']):
                return True
        return False
    
    def record_login_attempt(self, success):
        """Record login attempt success/failure."""
        if success:
            self.failed_login_attempts = 0
            self.last_failed_login = None
        else:
            self.failed_login_attempts += 1
            self.last_failed_login = timezone.now()
        self.save(update_fields=['failed_login_attempts', 'last_failed_login'])
    
    def is_temporarily_locked(self):
        """Check if account is temporarily locked due to failed attempts."""
        if self.failed_login_attempts >= settings.AXES_FAILURE_LIMIT:
            if self.last_failed_login:
                lockout_period = timezone.timedelta(hours=settings.AXES_COOLOFF_TIME)
                return timezone.now() < self.last_failed_login + lockout_period
        return False
    
    def set_security_questions(self, questions_and_answers):
        """Set security questions and answers for password recovery."""
        import bcrypt
        
        # Hash answers before storing
        hashed_qa = {}
        for question, answer in questions_and_answers.items():
            answer = answer.lower().strip()
            salt = bcrypt.gensalt()
            hashed_answer = bcrypt.hashpw(answer.encode(), salt)
            hashed_qa[question] = {
                'hash': hashed_answer.decode(),
                'salt': salt.decode()
            }
        
        self.security_questions = hashed_qa
        self.save(update_fields=['security_questions'])
    
    def check_security_answer(self, question, answer):
        """Verify a security question answer."""
        import bcrypt
        
        if question not in self.security_questions:
            return False
        
        stored = self.security_questions[question]
        answer = answer.lower().strip().encode()
        stored_hash = stored['hash'].encode()
        
        return bcrypt.checkpw(answer, stored_hash)

class XRayImage(models.Model):
    id: AutoField = models.AutoField(primary_key=True)
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
        ('N', 'Not Specified'),
    ]

    image = models.ImageField(
        upload_to='xray_images/',
        validators=[validate_image_extension, validate_image_size]
    )
    patient_name = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        default=None,
        validators=[MinLengthValidator(2)]
    )
    patient_surname = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        default=None,
        validators=[MinLengthValidator(2)]
    )
    patient_id = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        default=None,
        validators=[
            RegexValidator(
                regex=r'^[A-Za-z0-9-]+$',
                message='Patient ID can only contain letters, numbers, and hyphens'
            )
        ]
    )
    patient_date_of_birth = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        default=None
    )
    patient_gender = models.CharField(
        max_length=1,
        choices=GENDER_CHOICES,
        null=True,
        blank=True,
        default=None
    )
    xray_date = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        default=None
    )
    prediction = models.CharField(max_length=20, blank=True, default="No data")
    confidence = models.FloatField(default=0.0)
    normal_probability = models.FloatField(default=0.0)
    pneumonia_probability = models.FloatField(default=0.0)
    processing_time = models.FloatField(default=0.0)
    image_size = models.CharField(max_length=50, default="No data")
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_DEFAULT,
        default=get_default_user,
        related_name='xray_images'
    )

    class Meta:
        verbose_name = 'X-Ray Image'
        verbose_name_plural = 'X-Ray Images'
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['prediction']),
            models.Index(fields=['patient_id']),
            models.Index(fields=['uploaded_at']),
        ]

    def __str__(self):
        return f"XRay Image - {self.patient_name} {self.patient_surname} ({self.uploaded_at})"

    def get_extension(self):
        """Get the file extension without the dot."""
        name = self.image.name
        return name.split('.')[-1] if '.' in name else ''

    def get_file_size_mb(self):
        """Get the file size in MB with 2 decimal places."""
        try:
            if not self.image or not self.image.storage.exists(self.image.name):
                return "0.00"
            size_bytes = self.image.size
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f}"
        except Exception as e:
            logger.error(f"Error getting file size: {str(e)}")
            return "0.00"

    def clean(self):
        """Additional model validation."""
        if self.confidence and (self.confidence < 0 or self.confidence > 100):
            raise ValidationError({'confidence': 'Confidence must be between 0 and 100'})
        
        if self.normal_probability and (self.normal_probability < 0 or self.normal_probability > 100):
            raise ValidationError({'normal_probability': 'Probability must be between 0 and 100'})
        
        if self.pneumonia_probability and (self.pneumonia_probability < 0 or self.pneumonia_probability > 100):
            raise ValidationError({'pneumonia_probability': 'Probability must be between 0 and 100'})

    def save(self, *args, **kwargs):
        """Override save method to perform additional operations."""
        # Only title case the name and surname if they are not None
        if self.patient_name:
            self.patient_name = self.patient_name.title()
        if self.patient_surname:
            self.patient_surname = self.patient_surname.title()
        self.clean()
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Override delete method to ensure image file is deleted."""
        try:
            self.image.delete(save=False)
        except Exception as e:
            logger.error(f"Error deleting image file: {str(e)}")
        super().delete(*args, **kwargs)

    def get_formatted_dob(self):
        """Get formatted date of birth."""
        if not self.patient_date_of_birth:
            return _("No data")
        try:
            if isinstance(self.patient_date_of_birth, str):
                date_obj = datetime.strptime(self.patient_date_of_birth, '%Y-%m-%d')
            else:
                date_obj = self.patient_date_of_birth
            return date_obj.strftime('%Y-%m-%d')
        except (ValueError, AttributeError):
            return _("No data")

    def get_formatted_xray_date(self):
        """Get formatted X-ray date."""
        if not self.xray_date:
            return _("No data")
        try:
            if isinstance(self.xray_date, str):
                date_obj = datetime.strptime(self.xray_date, '%Y-%m-%d')
            else:
                date_obj = self.xray_date
            return date_obj.strftime('%Y-%m-%d')
        except (ValueError, AttributeError):
            return _("No data")

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
        logger.info(f"Initializing LungClassifier with device: {self.device}")

    def build_model(self):
        try:
            model = LungClassifierModel()
            model = model.to(self.device)
            self.model = model
            logger.info("Successfully built LungClassifierModel")
            return model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image.unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, image_path):
        try:
            # Build or load model if not already done
            if self.model is None:
                self.build_model()
                if not self.model:
                    logger.error("Failed to build model!")
                    raise RuntimeError("Failed to build model!")
                    
                if os.path.exists(settings.MODEL_PATH):
                    try:
                        state_dict = torch.load(settings.MODEL_PATH, map_location=self.device)
                        self.model.load_state_dict(state_dict)
                        logger.info("Successfully loaded model weights")
                    except Exception as e:
                        logger.error(f"Error loading model weights: {str(e)}")
                        raise RuntimeError(f"Failed to load model weights: {str(e)}")
                else:
                    logger.error("Model weights not found!")
                    raise FileNotFoundError("Model weights file not found!")

            # Get image size before preprocessing
            with Image.open(image_path) as img:
                original_size = f"{img.width}x{img.height}"

            # Start timing
            start_time = time.time()
            
            self.model.eval()
            with torch.no_grad():
                preprocessed_image = self.preprocess_image(image_path)
                logits = self.model(preprocessed_image)[0][0]
                prediction = torch.sigmoid(logits).item()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Convert prediction to class label and confidence
            pneumonia_prob = prediction
            normal_prob = 1 - prediction
            class_idx = 1 if pneumonia_prob > 0.5 else 0
            confidence = pneumonia_prob if pneumonia_prob > 0.5 else normal_prob
            
            result = {
                'class': self.classes[class_idx],
                'confidence': float(confidence * 100),
                'normal_probability': float(normal_prob * 100),
                'pneumonia_probability': float(pneumonia_prob * 100),
                'processing_time': float(processing_time),
                'image_size': original_size
            }
            
            logger.info(f"Successfully processed image. Prediction: {result['class']}, Confidence: {result['confidence']}%")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise