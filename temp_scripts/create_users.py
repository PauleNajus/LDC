import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')
django.setup()

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group

User = get_user_model()

def create_users():
    # Create admin user
    admin_user = User.objects.create_superuser(
        username='paubun',
        email='paubun@placeholder.com',
        password='pauliusbundza2025!',
        first_name='Paulius',
        last_name='Bundza',
    )
    print(f"Created admin user: {admin_user.username}")

    # Create regular user
    regular_user = User.objects.create_user(
        username='justri',
        email='justri@placeholder.com',
        password='justastrinkunas2025!',
        first_name='Justas',
        last_name='Trinkunas',
    )
    print(f"Created regular user: {regular_user.username}")

if __name__ == '__main__':
    create_users() 