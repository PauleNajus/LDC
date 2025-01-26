import os
import sys
from waitress import serve
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env.prod')

# Production settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings_prod')
os.environ['DJANGO_DEBUG'] = 'False'

# Import application after setting environment variables
from lung_classifier.wsgi import application
from django.conf import settings

def verify_production_settings():
    """Verify that we're running with production settings"""
    if settings.DEBUG:
        logger.error("WARNING: DEBUG is still True!")
        sys.exit(1)
    
    logger.info("Production settings verified:")
    logger.info(f"- DEBUG: {settings.DEBUG}")
    logger.info(f"- ALLOWED_HOSTS: {settings.ALLOWED_HOSTS}")
    logger.info(f"- STATIC_ROOT: {settings.STATIC_ROOT}")
    logger.info(f"- MEDIA_ROOT: {settings.MEDIA_ROOT}")
    logger.info(f"- DATABASE: {settings.DATABASES['default']['ENGINE']}")

if __name__ == '__main__':
    logger.info('Starting Lung Disease Classifier in production mode...')
    verify_production_settings()
    
    logger.info('Starting Waitress production server...')
    serve(
        application,
        host='localhost',
        port=8000,
        threads=6,
        url_scheme='http',
        channel_timeout=120,
        cleanup_interval=30,
        ident='Lung Disease Classifier'
    ) 