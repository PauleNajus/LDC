import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')
django.setup()

from django.contrib.auth.models import User

if not User.objects.filter(username='paubun').exists():
    User.objects.create_superuser('paubun', 'bundzapaulius@gmail.com', 'z#fjjEQKmQN%&YbkqT#z25qujEcL&H2A') 