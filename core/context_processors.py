from django.conf import settings
import torch
import platform
import psutil
import datetime

def system_info(request):
    """Add system information to the template context."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'system_info': {
                'python_version': platform.python_version(),
                'django_version': settings.DJANGO_VERSION,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'memory_used': f"{memory.percent}%",
                'disk_used': f"{disk.percent}%",
                'server_time': datetime.datetime.now(),
            }
        }
    except Exception:
        return {'system_info': {}}

def site_info(request):
    """Add site-wide information to the template context."""
    return {
        'site_info': {
            'title': 'Lung Disease Classifier',
            'description': 'AI-powered chest X-ray analysis system',
            'version': '1.0.0',
            'debug': settings.DEBUG,
            'contact_email': 'support@example.com',
        }
    }

def user_info(request):
    """Add user-specific information to the template context."""
    if request.user.is_authenticated:
        return {
            'user_info': {
                'full_name': request.user.get_full_name(),
                'email': request.user.email,
                'is_staff': request.user.is_staff,
                'last_login': request.user.last_login,
                'date_joined': request.user.date_joined,
            }
        }
    return {'user_info': {}}

def navigation(request):
    """Add navigation menu items to the template context."""
    menu_items = [
        {
            'title': 'Home',
            'url': 'core:home',
            'icon': 'fas fa-home',
            'active': request.resolver_match and request.resolver_match.url_name == 'home'
        },
        {
            'title': 'About',
            'url': 'core:about',
            'icon': 'fas fa-info-circle',
            'active': request.resolver_match and request.resolver_match.url_name == 'about'
        },
    ]
    
    if request.user.is_staff:
        menu_items.extend([
            {
                'title': 'Admin',
                'url': 'admin:index',
                'icon': 'fas fa-cog',
                'active': False
            },
        ])
    
    return {'navigation': menu_items} 