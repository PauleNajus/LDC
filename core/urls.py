from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('result/<int:prediction_id>/', views.result, name='result'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 