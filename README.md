# Lung Disease Classification System

A deep learning-based web application for classifying lung X-ray images as normal or pneumonia-affected using a PyTorch Convolutional Neural Network (CNN).

## Features

- Upload and classify chest X-ray images in popular formats (JPEG, PNG)
- Real-time classification using PyTorch CNN model
- Modern responsive UI with Bootstrap and dark/light mode
- Patient data management with secure authentication
- Performance metrics visualization
- Multi-language support

## Tech Stack

- **Backend:** Python 3.11+, Django 5.0.1
- **AI/ML:** PyTorch 2.1.2+, TorchVision, NumPy, OpenCV
- **Frontend:** Bootstrap 5, Font Awesome, JavaScript
- **Database:** SQLite (development), PostgreSQL (production ready)
- **Caching:** Redis
- **Security:** Django-CSP, Django-Axes, Argon2 password hashing

## System Requirements

- Python 3.11 or higher
- For GPU acceleration: NVIDIA GPU with CUDA support
- 8GB RAM (16GB+ recommended for training)
- 500MB free disk space (more for dataset storage)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lung-disease-classifier.git
cd lung-disease-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
python manage.py migrate
```

5. Create a superuser:
```bash
python manage.py createsuperuser
```

6. Run the development server:
```bash
python manage.py runserver
```

7. Access the application at http://127.0.0.1:8000

## Usage

1. Log in with your credentials
2. Navigate to the home page
3. Upload a chest X-ray image
4. View the classification results and confidence score
5. Manage patient details and previous predictions

## Model Architecture

The system uses a PyTorch CNN based on a customized DenseNet architecture with:
- Transfer learning from pre-trained weights
- Advanced normalization and regularization techniques
- Optimized for high accuracy on lung X-ray classification

## Training Your Own Model

The system includes a comprehensive training module (`core/train_model.py`) with:
- Data augmentation using Albumentations
- Mixed precision training
- Learning rate scheduling
- Early stopping and model checkpointing
- Cross-validation and metrics reporting

### Dataset Structure
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

### Run Training
```bash
python core/train_model.py --data_dir path/to/dataset
```

## Production Deployment

For production deployment, the project includes:
- Gunicorn configuration
- Nginx setup files
- WSGI for Windows (Waitress)
- Production settings with security enhancements
- Environment variable management

## License

MIT License 