import os
import django

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

def change_passwords():
    # Change password for paubun
    try:
        paubun = User.objects.get(username='paubun')
        paubun.set_password('PauBun2025!')
        paubun.save()
        print(f"Password changed for user: {paubun.username}")
    except User.DoesNotExist:
        print("User 'paubun' not found.")
    except Exception as e:
        print(f"Error changing password for paubun: {str(e)}")
    
    # Change password for justri
    try:
        justri = User.objects.get(username='justri')
        justri.set_password('JusTri2025!')
        justri.save()
        print(f"Password changed for user: {justri.username}")
    except User.DoesNotExist:
        print("User 'justri' not found.")
    except Exception as e:
        print(f"Error changing password for justri: {str(e)}")

if __name__ == '__main__':
    change_passwords() 