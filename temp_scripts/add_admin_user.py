import os
import django

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

def create_admin_user():
    # Check if the user already exists
    if User.objects.filter(username='admin').exists():
        print("Admin user already exists!")
        return
    
    # Create admin user with the specified details
    admin_user = User.objects.create_superuser(
        username='admin',
        email='admin@gmail.com',
        password='Admin2025!',
        first_name='adminfirst',
        last_name='adminlast',
    )
    print(f"Created admin user: {admin_user.username}")
    print(f"Email: {admin_user.email}")
    print(f"Name: {admin_user.first_name} {admin_user.last_name}")
    print(f"Is superuser: {admin_user.is_superuser}")

if __name__ == '__main__':
    create_admin_user() 