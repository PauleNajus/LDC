import os
import sys
import platform
import psutil
import torch
import django
from django.core.management.base import BaseCommand
from django.conf import settings
import logging

logger = logging.getLogger('django')

class Command(BaseCommand):
    help = 'Check and log system status, including Python, Django, and PyTorch configurations'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Checking system status...'))
        
        # System Information
        logger.info('=== System Information ===')
        logger.info(f'OS: {platform.system()} {platform.version()}')
        logger.info(f'Python Version: {sys.version}')
        logger.info(f'Django Version: {django.get_version()}')
        
        # Memory Information
        memory = psutil.virtual_memory()
        logger.info('\n=== Memory Status ===')
        logger.info(f'Total Memory: {memory.total / (1024**3):.2f} GB')
        logger.info(f'Available Memory: {memory.available / (1024**3):.2f} GB')
        logger.info(f'Memory Usage: {memory.percent}%')
        
        # CPU Information
        logger.info('\n=== CPU Information ===')
        logger.info(f'CPU Cores: {psutil.cpu_count()}')
        logger.info(f'CPU Usage: {psutil.cpu_percent(interval=1)}%')
        
        # PyTorch Information
        logger.info('\n=== PyTorch Configuration ===')
        logger.info(f'PyTorch Version: {torch.__version__}')
        logger.info(f'CUDA Available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            logger.info(f'CUDA Version: {torch.version.cuda}')
            logger.info(f'GPU Device: {torch.cuda.get_device_name(0)}')
            logger.info(f'GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
            logger.info(f'GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
            logger.info(f'GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
        
        # Django Settings
        logger.info('\n=== Django Configuration ===')
        logger.info(f'Debug Mode: {settings.DEBUG}')
        logger.info(f'Database Engine: {settings.DATABASES["default"]["ENGINE"]}')
        logger.info(f'Static Root: {settings.STATIC_ROOT}')
        logger.info(f'Media Root: {settings.MEDIA_ROOT}')
        
        # Installed Apps
        logger.info('\n=== Installed Apps ===')
        for app in settings.INSTALLED_APPS:
            logger.info(f'- {app}')
        
        self.stdout.write(self.style.SUCCESS('System status check completed successfully')) 