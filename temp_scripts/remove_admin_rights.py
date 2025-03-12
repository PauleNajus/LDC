import os
import django

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

def remove_admin_rights():
    try:
        # Find the user
        user = User.objects.get(username='paubun')
        
        # Check if the user has admin rights
        if user.is_superuser:
            # Remove admin rights
            user.is_superuser = False
            user.is_staff = False  # Also remove staff status which is needed for admin access
            user.save()
            print(f"Admin rights removed from user: {user.username}")
            print(f"Superuser status: {user.is_superuser}")
            print(f"Staff status: {user.is_staff}")
        else:
            print(f"User {user.username} doesn't have admin rights.")
            
    except User.DoesNotExist:
        print("User 'paubun' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    remove_admin_rights() 