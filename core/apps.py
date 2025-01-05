from django.apps import AppConfig
from django.core.cache import cache

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    verbose_name = 'Lung Disease Classifier'

    def ready(self):
        """Initialize app and register signals."""
        try:
            # Import signals
            import core.signals
            
            # Clear all cache on startup
            cache.clear()
            
        except Exception as e:
            import logging
            logger = logging.getLogger('core')
            logger.error(f'Error initializing app: {str(e)}') 