import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

class LungClassifier:
    def __init__(self):
        self.model = None
        self.model_path = Path(__file__).parent / 'models' / 'lung_model.h5'
        self.image_size = (224, 224)
        self.load_model()

    def load_model(self):
        """Load the trained model if it exists."""
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a basic model if the trained one doesn't exist
            self.model = self._create_basic_model()

    def _create_basic_model(self):
        """Create a basic CNN model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*self.image_size, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_image(self, image):
        """Preprocess the image for model input."""
        # Convert uploaded file to numpy array
        if hasattr(image, 'read'):
            # Read the file into a numpy array
            image_array = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # If it's already a path or numpy array
            img = cv2.imread(str(image))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.image_size)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img

    def predict(self, image):
        """Predict whether the X-ray shows pneumonia or not."""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(processed_image)[0][0]
            
            # Convert prediction to label and confidence
            label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
            confidence = float(prediction if prediction > 0.5 else 1 - prediction) * 100
            
            return label, confidence
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "ERROR", 0.0 