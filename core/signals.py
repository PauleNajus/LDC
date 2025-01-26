from django.db.models.signals import post_save, post_delete, pre_delete
from django.dispatch import receiver
from django.core.cache import cache
from django.conf import settings
from .models import XRayImage, User
import logging
import os
from django.contrib.auth import get_user_model

logger = logging.getLogger('core')
User = get_user_model()

@receiver(post_save, sender=XRayImage)
def handle_new_prediction(sender, instance, created, **kwargs):
    """Handle actions when a new prediction is created."""
    if created:
        try:
            # Log the new prediction
            logger.info(
                f'New prediction created: ID={instance.id}, '
                f'Patient={instance.patient_name} {instance.patient_surname}, '
                f'Result={instance.prediction}'
            )
            
            # Clear any cached predictions for this user
            cache_key = f'user_predictions_{instance.user.id}'
            cache.delete(cache_key)
            
            # Update user's last activity
            if instance.user:
                instance.user.last_login = instance.uploaded_at
                instance.user.save(update_fields=['last_login'])
            
        except Exception as e:
            logger.error(f'Error handling new prediction: {str(e)}')

@receiver(pre_delete, sender=XRayImage)
def handle_prediction_pre_delete(sender, instance, **kwargs):
    """Handle actions before a prediction is deleted."""
    try:
        # Log the deletion
        logger.info(
            f'Deleting prediction: ID={instance.id}, '
            f'Patient={instance.patient_name} {instance.patient_surname}, '
            f'Result={instance.prediction}'
        )
        
        # Clear any cached predictions for this user
        if instance.user:
            cache_key = f'user_predictions_{instance.user.id}'
            cache.delete(cache_key)
        
    except Exception as e:
        logger.error(f'Error handling prediction pre-delete: {str(e)}')

@receiver(post_delete, sender=XRayImage)
def handle_prediction_post_delete(sender, instance, **kwargs):
    """Handle actions after a prediction is deleted."""
    try:
        # Clean up any orphaned image files
        if instance.image:
            image_path = instance.image.path
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info(f'Deleted image file: {image_path}')
                except Exception as e:
                    logger.error(f'Error deleting image file {image_path}: {str(e)}')
        
        # Clean up empty directories in the media folder
        media_root = settings.MEDIA_ROOT
        for root, dirs, files in os.walk(media_root, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # if directory is empty
                        os.rmdir(dir_path)
                        logger.info(f'Deleted empty directory: {dir_path}')
                except Exception as e:
                    logger.error(f'Error deleting directory {dir_path}: {str(e)}')
        
    except Exception as e:
        logger.error(f'Error handling prediction post-delete: {str(e)}')

@receiver(post_save, sender=User)
def user_updated(sender, instance, created, **kwargs):
    """Signal handler for user model updates."""
    try:
        logger.info(f"User updated: ID={instance.id}, Username={instance.username}, Email={instance.email}")
        
        # Clear user-specific cache
        cache_keys = [
            f'user_{instance.id}_profile',
            f'user_{instance.id}_permissions',
            f'user_{instance.username}_data'
        ]
        cache.delete_many(cache_keys)
        
    except Exception as e:
        logger.error(f"Error handling user update: {str(e)}") 