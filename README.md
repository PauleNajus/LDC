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

## Production Deployment

### Prerequisites
- Python 3.10 or higher
- PostgreSQL
- Redis
- Nginx
- Git

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lung-disease-classifier.git
cd lung-disease-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the production environment template
cp .env.prod .env

# Edit the .env file with your production settings
nano .env
```

4. Create necessary directories:
```bash
mkdir logs mediafiles staticfiles
```

5. Set up the database:
```bash
python manage.py migrate
python manage.py createsuperuser
```

6. Collect static files:
```bash
python manage.py collectstatic --no-input
```

### Production Server Setup

1. Configure Nginx:
```bash
# Copy the Nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/lung_classifier
sudo ln -s /etc/nginx/sites-available/lung_classifier /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

2. Set up SSL (recommended):
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

3. Start the Gunicorn server:
```bash
gunicorn -c gunicorn_config.py lung_classifier.wsgi:application
```

### Running in Production

#### Windows
1. Install Waitress:
```bash
pip install waitress
```

2. Start the production server:
```bash
python run_prod_windows.py
```

The server will be available at http://localhost:8000

#### Linux/Unix
1. Create a systemd service file:
```bash
sudo nano /etc/systemd/system/lung_classifier.service
```

2. Add the following content:
```ini
[Unit]
Description=Lung Disease Classifier Gunicorn Daemon
After=network.target

[Service]
User=your_user
Group=your_group
WorkingDirectory=/path/to/lung_classifier
Environment="PATH=/path/to/virtualenv/bin"
EnvironmentFile=/path/to/lung_classifier/.env
ExecStart=/path/to/virtualenv/bin/gunicorn -c gunicorn_config.py lung_classifier.wsgi:application

[Install]
WantedBy=multi-user.target
```

3. Start and enable the service:
```bash
sudo systemctl start lung_classifier
sudo systemctl enable lung_classifier
```

### Monitoring and Maintenance

1. View logs:
```bash
# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Application logs
tail -f logs/django.log
tail -f logs/access.log
tail -f logs/error.log
```

2. Monitor system resources:
```bash
htop  # CPU and memory usage
df -h  # Disk usage
```

3. Backup database:
```bash
pg_dump lung_classifier_db > backup_$(date +%Y%m%d).sql
```

### Security Considerations

1. Keep the system updated:
```bash
sudo apt update && sudo apt upgrade
pip install --upgrade -r requirements.txt
```

2. Check security headers:
```bash
curl -I https://your-domain.com
```

3. Regular maintenance:
- Monitor error logs and Sentry dashboard
- Keep backups current
- Rotate logs
- Update SSL certificates
- Monitor system resources

### Troubleshooting

1. Check application status:
```bash
sudo systemctl status lung_classifier
```

2. View logs for errors:
```bash
sudo journalctl -u lung_classifier
```

3. Test Nginx configuration:
```bash
sudo nginx -t
```

4. Check permissions:
```bash
sudo chown -R www-data:www-data mediafiles staticfiles
sudo chmod -R 755 mediafiles staticfiles
```

For more information or troubleshooting, please refer to the official documentation:
- [Django Deployment Checklist](https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/)
- [Gunicorn Documentation](https://docs.gunicorn.org/en/stable/configure.html)
- [Nginx Documentation](https://nginx.org/en/docs/) 