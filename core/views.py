from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.core.exceptions import ValidationError
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import logout, get_user_model
from django.urls import reverse_lazy
from django.views.generic import View, TemplateView, ListView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import Paginator
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.utils.translation import gettext as _
from django.core.cache import cache
from django.conf import settings
from django.http import Http404, JsonResponse
from .models import XRayImage, LungClassifier
from .forms import XRayImageForm
from datetime import datetime
import logging
import time
from django.utils import translation
from django.urls import reverse
from django.conf import global_settings
import torch

logger = logging.getLogger('core')

def format_date(date_str):
    """Format date to Lithuanian format or return 'No data' if empty"""
    if not date_str:
        return "No data"
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return "No data"

class CustomLoginView(LoginView):
    template_name = 'core/login.html'
    redirect_authenticated_user = True

    def form_valid(self, form):
        user = form.get_user()
        logger.info(f"Successful login for user: {user.get_username()}")
        return super().form_valid(form)

    def form_invalid(self, form):
        if username := form.cleaned_data.get('username'):
            logger.warning(f"Failed login attempt for user: {username}")
        return super().form_invalid(form)

class CustomLogoutView(LogoutView):
    next_page = 'core:home'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            logger.info(f"User logged out: {request.user.get_username()}")
        return super().dispatch(request, *args, **kwargs)

class HomeView(LoginRequiredMixin, TemplateView):
    template_name = 'core/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get recent predictions with pagination
        recent_predictions = XRayImage.objects.filter(user=self.request.user).order_by('-uploaded_at')
        paginator = Paginator(recent_predictions, 10)
        page = self.request.GET.get('page')
        context['recent_predictions'] = paginator.get_page(page)
        context['form'] = XRayImageForm()
        
        return context
    
    def post(self, request, *args, **kwargs):
        logger.info('\n=== Starting Image Analysis Process ===')
        logger.info(f'Timestamp: {timezone.now()}')
        logger.info(f'User: {request.user.get_username()}')
        logger.info(f'Request Method: {request.method}')
        logger.info(f'Content Type: {request.content_type}')
        
        # Log request details
        if request.FILES:
            image_file = request.FILES.get('image')
            if image_file:
                logger.info('\n=== Image Details ===')
                logger.info(f'Original Filename: {image_file.name}')
                logger.info(f'File Size: {image_file.size / 1024:.2f} KB')
                logger.info(f'Content Type: {image_file.content_type}')
        
        # Log form data (excluding sensitive information)
        logger.info('\n=== Form Data ===')
        safe_fields = ['patient_id', 'xray_date']
        for field in safe_fields:
            value = request.POST.get(field)
            if value:
                logger.info(f'{field}: {value}')
        
        form = XRayImageForm(request.POST, request.FILES)
        logger.info(f'\nForm is valid: {form.is_valid()}')
        
        if form.is_valid():
            try:
                logger.info('\n=== Processing Image ===')
                # Create XRayImage instance but don't save yet
                xray = form.save(commit=False)
                xray.user = request.user
                
                # Get or create classifier instance from cache
                classifier_key = 'lung_classifier'
                classifier = cache.get(classifier_key)
                if not classifier:
                    logger.info('Creating new classifier instance (cache miss)')
                    classifier = LungClassifier()
                    cache.set(classifier_key, classifier, timeout=3600)
                else:
                    logger.info('Using cached classifier instance')
                
                # Process image and make prediction
                logger.info('Starting prediction process...')
                start_time = time.time()
                
                # Get model output
                model_output = classifier.predict(xray.image)
                processing_time = time.time() - start_time
                
                # Log prediction results
                logger.info('\n=== Prediction Results ===')
                logger.info(f'Processing Time: {processing_time:.3f} seconds')
                
                # Handle model output
                if isinstance(model_output, dict):
                    # If output is already processed
                    predicted_class = model_output.get('class', 0)
                    confidence = float(model_output.get('confidence', 0.0))
                    normal_prob = float(model_output.get('normal_probability', 0.0))
                    pneumonia_prob = float(model_output.get('pneumonia_probability', 0.0))
                else:
                    # Process raw tensor output
                    logits = model_output.squeeze()
                    probs = torch.nn.functional.softmax(logits, dim=0)
                    predicted_class = int(torch.argmax(probs).item())
                    
                    # Convert probabilities to percentages (0-100 range)
                    normal_prob = float(probs[0].item()) * 100
                    pneumonia_prob = float(probs[1].item()) * 100
                    confidence = normal_prob if predicted_class == 0 else pneumonia_prob
                
                logger.info(f'Predicted Class: {predicted_class}')
                logger.info(f'Confidence: {confidence:.4f}%')
                logger.info(f'Normal Probability: {normal_prob:.4f}%')
                logger.info(f'Pneumonia Probability: {pneumonia_prob:.4f}%')
                
                # Update XRayImage with prediction results
                prediction_class = 'NORMAL' if normal_prob > pneumonia_prob else 'PNEUMONIA'
                xray.prediction = prediction_class
                xray.confidence = max(normal_prob, pneumonia_prob)  # Use the higher probability as confidence
                xray.normal_probability = normal_prob
                xray.pneumonia_probability = pneumonia_prob
                xray.processing_time = processing_time
                xray.image_size = f"{xray.image.width}x{xray.image.height}"
                
                # Save the instance
                logger.info('\n=== Saving Results ===')
                xray.save()
                logger.info(f'Saved XRayImage with ID: {xray.id}')
                
                logger.info(f"\nSuccessfully processed X-ray image for user {request.user.get_username()}")
                messages.success(request, _("X-ray image processed successfully."))
                
                # Return JSON response for AJAX request
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'success': True})
                
                # Stay on home page instead of redirecting to result
                return redirect('core:home')
                
            except Exception as e:
                logger.error('\n=== Error Processing Image ===')
                logger.error(f'Error Type: {type(e).__name__}')
                logger.error(f'Error Message: {str(e)}')
                logger.error(f'Error Details:', exc_info=True)
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'success': False,
                        'errors': str(e)
                    }, status=500)
                
                messages.error(request, _("Error processing X-ray image. Please try again."))
                return self.render_to_response(self.get_context_data(form=form))
        else:
            logger.warning('\n=== Form Validation Failed ===')
            logger.warning('Form Errors:')
            errors = {}
            for field, error_list in form.errors.items():
                errors[field] = [str(error) for error in error_list]
                logger.warning(f'{field}: {", ".join(errors[field])}')
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'errors': errors
                }, status=400)
            
            messages.error(request, _("Please correct the errors below."))
            return self.render_to_response(self.get_context_data(form=form))

class AboutView(TemplateView):
    template_name = 'core/about.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Cache the about page content for 15 minutes
        cache_key = 'about_page_content'
        content = cache.get(cache_key)
        
        if not content:
            content = {
                'model_architecture': 'CNN with ResNet backbone',
                'training_dataset': 'Chest X-Ray Images (Pneumonia)',
                'accuracy': '95.3%',
                'last_updated': timezone.now(),
            }
            cache.set(cache_key, content, timeout=900)
        
        context.update(content)
        return context

class ResultView(LoginRequiredMixin, TemplateView):
    template_name = 'core/result.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        prediction_id = self.kwargs.get('prediction_id')
        
        try:
            prediction = get_object_or_404(XRayImage, id=prediction_id)
            
            # Check if user has permission to view this result
            # Use user.is_authenticated and getattr for is_staff check
            if (prediction.user != self.request.user and 
                not (self.request.user.is_authenticated and getattr(self.request.user, 'is_staff', False))):
                raise Http404(_("You do not have permission to view this result."))
            
            context['prediction'] = prediction
            
        except Exception as e:
            logger.error(f"Error retrieving prediction {prediction_id}: {str(e)}")
            messages.error(self.request, _("Error retrieving prediction results."))
        
        return context

class DeletePredictionView(LoginRequiredMixin, View):
    def post(self, request, prediction_id):
        try:
            prediction = get_object_or_404(XRayImage, id=prediction_id)
            
            # Check if user has permission to delete this prediction
            if prediction.user != request.user and not request.user.is_staff:
                messages.error(request, _("You do not have permission to delete this prediction."))
                return redirect('core:home')
            
            # Delete the prediction
            prediction.delete()
            messages.success(request, _("Prediction deleted successfully."))
            logger.info(f"User {request.user.username} deleted prediction {prediction_id}")
            
        except Exception as e:
            logger.error(f"Error deleting prediction {prediction_id}: {str(e)}")
            messages.error(request, _("Error deleting prediction."))
        
        return redirect('core:home')

def set_language(request):
    """View to handle language switching."""
    if request.method == 'POST':
        language = request.POST.get('language')
        next_url = request.POST.get('next', '/')
        
        if language and language in dict(settings.LANGUAGES):
            # Update user's language preference if authenticated
            if request.user.is_authenticated:
                request.user.language_preference = language
                request.user.save()
            
            # Set language in session and activate it
            translation.activate(language)
            request.session[translation.LANGUAGE_SESSION_KEY] = language
            
            # Handle URL prefix
            path_parts = next_url.lstrip('/').split('/')
            if path_parts and path_parts[0] in dict(settings.LANGUAGES):
                # Remove current language prefix
                path_parts = path_parts[1:]
            
            # Construct new URL with language prefix
            next_url = f'/{language}/{"/".join(path_parts)}'
            
            # Create response with redirect
            response = redirect(next_url)
            
            # Set language cookie
            response.set_cookie(
                settings.LANGUAGE_COOKIE_NAME,
                language,
                max_age=settings.LANGUAGE_COOKIE_AGE,
                path=settings.LANGUAGE_COOKIE_PATH,
                domain=settings.LANGUAGE_COOKIE_DOMAIN,
                secure=settings.LANGUAGE_COOKIE_SECURE,
                httponly=settings.LANGUAGE_COOKIE_HTTPONLY,
                samesite=settings.LANGUAGE_COOKIE_SAMESITE,
            )
            
            return response
    
    return redirect('core:home')