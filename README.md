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
- NVIDIA GPU with CUDA support (RTX 4080 or better recommended)
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

## Model Training

### Dataset Preparation
1. Organize your chest X-ray dataset in the following structure:
```
dataset/
├── train/
│   ├── normal/
│   │   └── (normal X-ray images)
│   └── pneumonia/
│       └── (pneumonia X-ray images)
├── val/
│   ├── normal/
│   │   └── (normal X-ray images)
│   └── pneumonia/
│       └── (pneumonia X-ray images)
└── test/
    ├── normal/
    │   └── (normal X-ray images)
    └── pneumonia/
        └── (pneumonia X-ray images)
```

2. Ensure your images are in JPEG or PNG format and properly labeled.

### Training Configuration
The training script (`core/train_model.py`) includes the following optimized parameters:
- Batch size: 128 (optimized for RTX 4080 12GB)
- Number of workers: 12
- Early stopping patience: 15 epochs
- Minimum epochs: 50
- Maximum epochs: 200
- Target accuracy: 98%
- 5-fold cross-validation

### Hardware Requirements
- NVIDIA GPU with at least 12GB VRAM (RTX 4080 or better recommended)
- CUDA toolkit installed and configured
- 64GB RAM recommended for optimal performance
- High-speed storage for dataset loading

### Running the Training
1. Activate your virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the training script:
```bash
python core/train_model.py --data_dir path/to/dataset
```

3. Monitor the training progress:
- Training metrics will be displayed in real-time
- Model checkpoints will be saved in the `models/` directory
- Training history plots will be saved in the `static/` directory

### Training Features
- Mixed precision training for optimal GPU utilization
- Advanced data augmentation using Albumentations
- Gradient accumulation for stable training
- Learning rate scheduling
- Early stopping to prevent overfitting
- Model checkpointing for best weights
- Cross-validation for robust evaluation

### Post-Training
After training completes:
1. The best model will be saved as `models/best_model.pth`
2. Training history plots will be available in `static/training_history.png`
3. Cross-validation results will be saved in `models/cv_results.json`

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