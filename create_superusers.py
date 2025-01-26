import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')
django.setup()

from core.models import User

def create_superusers():
    # Create first superuser
    if not User.objects.filter(username='paubun').exists():
        User.objects.create_superuser(
            username='paubun',
            password='paubun123',
            email='paulius.bundza@santa.lt',
            first_name='Paulius',
            last_name='Bundza'
        )
        print("Created superuser: paubun")

    # Create second superuser
    if not User.objects.filter(username='justri').exists():
        User.objects.create_superuser(
            username='justri',
            password='justri123',
            email='justas.trinkunas@santa.lt',
            first_name='Justas',
            last_name='Trinkunas'
        )
        print("Created superuser: justri")

if __name__ == '__main__':
    create_superusers() 