from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
from core.models import XRayImage
import logging
import datetime

logger = logging.getLogger('core')

class Command(BaseCommand):
    help = 'Clean up old predictions and their associated files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Delete predictions older than this many days'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )

    def handle(self, *args, **options):
        days = options['days']
        dry_run = options['dry_run']
        
        # Calculate the cutoff date
        cutoff_date = timezone.now() - datetime.timedelta(days=days)
        
        # Get predictions older than the cutoff date
        old_predictions = XRayImage.objects.filter(uploaded_at__lt=cutoff_date)
        count = old_predictions.count()
        
        if count == 0:
            self.stdout.write(
                self.style.SUCCESS(f'No predictions older than {days} days found.')
            )
            return
        
        # Log the start of the cleanup
        logger.info(f'Starting cleanup of predictions older than {days} days')
        logger.info(f'Found {count} predictions to delete')
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING(f'Would delete {count} predictions (dry run)')
            )
            for prediction in old_predictions:
                self.stdout.write(f'Would delete: {prediction}')
            return
        
        try:
            # Delete the predictions
            deleted_count = 0
            for prediction in old_predictions:
                try:
                    # Get prediction details for logging
                    details = {
                        'id': prediction.id,
                        'patient_id': prediction.patient_id,
                        'uploaded_at': prediction.uploaded_at,
                        'prediction': prediction.prediction,
                    }
                    
                    # Delete the prediction (this will also delete the image file
                    # due to our model's delete() method)
                    prediction.delete()
                    deleted_count += 1
                    
                    # Log successful deletion
                    logger.info(f'Successfully deleted prediction: {details}')
                    
                except Exception as e:
                    logger.error(f'Error deleting prediction {prediction.id}: {str(e)}')
                    continue
            
            # Log completion
            logger.info(f'Cleanup completed. Deleted {deleted_count} predictions')
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully deleted {deleted_count} predictions')
            )
            
        except Exception as e:
            logger.error(f'Error during cleanup: {str(e)}')
            self.stdout.write(
                self.style.ERROR(f'Error during cleanup: {str(e)}')
            ) 