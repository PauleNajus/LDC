# Lung Disease Classification System

A deep learning-based web application for classifying lung X-ray images as healthy or pneumonia-affected using Convolutional Neural Networks (CNN).

## Features
- Upload and classify chest X-ray images
- Real-time classification using CNN
- Modern UI with Tailwind CSS
- Detailed classification results
- Model performance metrics

## Tech Stack
- Python 3.11.9
- Django 5.1.4
- TensorFlow 2.15.0
- Tailwind CSS
- OpenCV
- Scikit-learn

## System Requirements
- Python 3.11.9
- NVIDIA GPU with CUDA support
- 64GB RAM recommended
- Windows 11/Linux/MacOS

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize Django project:
```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py tailwind install
python manage.py tailwind build
```

4. Run the development server:
```bash
python manage.py runserver
```

## Project Structure
- `/lung_classifier` - Main Django project
- `/core` - Main application
- `/models` - CNN model architecture and training
- `/templates` - HTML templates
- `/static` - Static files (CSS, JS, images)
- `/media` - Uploaded images

## Model Architecture
- CNN based on state-of-the-art architecture
- Transfer learning using pre-trained weights
- Data augmentation for better generalization
- Dropout and batch normalization for regularization

## License
MIT License 