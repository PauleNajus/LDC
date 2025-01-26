from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
from core.models import XRayImage
import logging
import datetime
import os

logger = logging.getLogger('core')

class Command(BaseCommand):
    help = 'Clean up predictions with missing image files'

    def handle(self, *args, **options):
        self.stdout.write('Starting cleanup of predictions with missing image files...')
        
        # Get all XRayImage records
        predictions = XRayImage.objects.all()
        total_count = predictions.count()
        deleted_count = 0
        
        for prediction in predictions:
            if not prediction.image or not prediction.image.storage.exists(prediction.image.name):
                # Log the deletion
                logger.info(f'Deleting prediction {prediction.id} due to missing image file: {prediction.image.name}')
                
                # Delete the record
                prediction.delete()
                deleted_count += 1
        
        # Print summary
        self.stdout.write(self.style.SUCCESS(
            f'Cleanup complete. Processed {total_count} predictions, deleted {deleted_count} records with missing files.'
        )) 