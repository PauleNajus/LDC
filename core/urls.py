from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'core'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('about/', views.AboutView.as_view(), name='about'),
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', views.CustomLogoutView.as_view(), name='logout'),
    path('result/<int:prediction_id>/', views.ResultView.as_view(), name='result'),
    path('delete/<int:prediction_id>/', views.DeletePredictionView.as_view(), name='delete_prediction'),
    path('update/<int:prediction_id>/', views.UpdatePredictionView.as_view(), name='update_prediction'),
    path('password_reset/', auth_views.PasswordResetView.as_view(
        template_name='core/password_reset.html',
        email_template_name='core/password_reset_email.html',
        subject_template_name='core/password_reset_subject.txt'
    ), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='core/password_reset_done.html'
    ), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='core/password_reset_confirm.html'
    ), name='password_reset_confirm'),
    path('password_reset/complete/', auth_views.PasswordResetCompleteView.as_view(
        template_name='core/password_reset_complete.html'
    ), name='password_reset_complete'),
    # X-Ray views
    path('xrays/', views.XRayListView.as_view(), name='xray_list'),
    path('xrays/<int:pk>/', views.XRayDetailView.as_view(), name='xray_detail'),
    # API endpoints
    path('api/test-prediction/', views.test_prediction, name='test_prediction'),
    path('api-test/', views.api_test_view, name='api_test_view'),
]

# Error handlers should be defined in the main urls.py file, not here
# handler404 = views.custom_404
# handler500 = views.custom_500 