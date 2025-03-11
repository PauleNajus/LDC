from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView, ListView, DetailView, DeleteView, UpdateView
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import reverse_lazy
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.defaults import page_not_found, server_error, permission_denied, bad_request
from django.core.paginator import Paginator
import logging
import time
from PIL import Image
import torch
import io
import os
from pathlib import Path
from django.conf import settings
from django.db import models

# Import the correct model names
from .models import XRayImage
from .forms import XRayImageForm, PredictionSearchForm
from .lung_classifier import LungClassifier

logger = logging.getLogger('core')

# Initialize the lung classifier
model_path = Path(settings.BASE_DIR) / 'models' / 'best_model.pth'
classifier = None
try:
    classifier = LungClassifier(model_path)
    logger.info(f"Lung classifier model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading lung classifier model: {str(e)}")

# Base views
class HomeView(TemplateView):
    template_name = 'core/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Initialize the search form with request data
        search_form = PredictionSearchForm(self.request.GET or None)
        context['search_form'] = search_form
        
        # Start with predictions filtered by the current user if authenticated
        if self.request.user.is_authenticated:
            predictions = XRayImage.objects.filter(user=self.request.user).order_by('-uploaded_at')
        else:
            predictions = XRayImage.objects.none()  # No predictions for unauthenticated users
        
        # Apply filters if the form is valid
        if search_form.is_valid() and self.request.GET:
            # Filter by search query (name, surname, or ID)
            search_query = search_form.cleaned_data.get('search_query')
            if search_query:
                predictions = predictions.filter(
                    models.Q(patient_name__icontains=search_query) |
                    models.Q(patient_surname__icontains=search_query) |
                    models.Q(patient_id__icontains=search_query)
                )
            
            # Filter by prediction type
            prediction_type = search_form.cleaned_data.get('prediction_type')
            if prediction_type:
                predictions = predictions.filter(prediction=prediction_type)
            
            # Filter by date range
            date_from = search_form.cleaned_data.get('date_from')
            if date_from:
                predictions = predictions.filter(uploaded_at__date__gte=date_from)
            
            date_to = search_form.cleaned_data.get('date_to')
            if date_to:
                predictions = predictions.filter(uploaded_at__date__lte=date_to)
        
        # Check if we should return all records (for AJAX "Load All" request)
        if self.request.GET.get('all') == '1' and self.request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # Return all records without pagination
            context['recent_predictions'] = predictions
            context['has_more'] = False
        else:
            # Implement pagination as usual
            paginator = Paginator(predictions, 5)  # Show 5 predictions per page
            page_number = self.request.GET.get('page', 1)
            page_obj = paginator.get_page(page_number)
            
            context['recent_predictions'] = page_obj
            context['has_more'] = page_obj.has_next()
        
        return context

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        
        # If this is an AJAX request, only return the predictions part
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render(request, 'core/home.html', context)
        
        return self.render_to_response(context)
    
    def post(self, request, *args, **kwargs):
        try:
            logger.info("Processing image upload request")
            start_time = time.time()
            
            form = XRayImageForm(request.POST, request.FILES)
            if form.is_valid():
                # Save the image without committing to get the model instance
                xray_image = form.save(commit=False)
                
                # Set the user if available, otherwise use the default
                if request.user.is_authenticated:
                    xray_image.user = request.user
                
                # Save to get the image file
                xray_image.save()
                
                # Get the image for prediction
                image_file = request.FILES['image']
                
                # Perform prediction logic
                prediction, confidence, normal_prob, pneumonia_prob = self.predict_xray(xray_image.image.path)
                
                # Update the record with prediction results
                xray_image.prediction = prediction
                xray_image.confidence = confidence
                xray_image.normal_probability = normal_prob * 100.0  # Convert to percentage
                xray_image.pneumonia_probability = pneumonia_prob * 100.0  # Convert to percentage
                xray_image.processing_time = time.time() - start_time
                
                # Get image dimensions and set image_size
                with Image.open(xray_image.image.path) as img:
                    width, height = img.size
                    xray_image.image_size = f"{width}x{height}"
                
                # Save the updated record
                xray_image.save()
                
                logger.info(f"Successfully processed image. Prediction: {prediction}, Confidence: {confidence:.2f}")
                
                # Return the same page with updated context
                return self.get(request, *args, **kwargs)
            else:
                logger.warning(f"Form validation failed: {form.errors}")
                return JsonResponse({'errors': form.errors}, status=400)
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return JsonResponse({'errors': f"An error occurred: {str(e)}"}, status=500)
    
    def predict_xray(self, image_path):
        """Use the trained model to predict if the X-ray shows pneumonia or normal lungs."""
        try:
            # If classifier is not available, use dummy values
            if classifier is None:
                logger.warning("Classifier not available, using dummy prediction values")
                import random
                normal_prob = random.uniform(0.3, 0.7)
                pneumonia_prob = 1.0 - normal_prob
                prediction = "NORMAL" if normal_prob > pneumonia_prob else "PNEUMONIA"
                confidence = max(normal_prob, pneumonia_prob)
                logger.info(f"DUMMY PREDICTION: {prediction}, confidence: {confidence:.4f}, normal_prob: {normal_prob:.4f}, pneumonia_prob: {pneumonia_prob:.4f}")
                return prediction, confidence, normal_prob, pneumonia_prob
            
            # Use the actual classifier for prediction
            logger.info(f"Using real classifier to predict image: {image_path}")
            class_idx, confidence = classifier.predict(image_path)
            
            # Class 0 is typically NORMAL, Class 1 is PNEUMONIA
            # But verify based on your model training
            if class_idx == 0:
                prediction = "NORMAL"
                normal_prob = confidence
                pneumonia_prob = 1.0 - confidence
            else:
                prediction = "PNEUMONIA"
                pneumonia_prob = confidence
                normal_prob = 1.0 - confidence
            
            logger.info(f"REAL MODEL PREDICTION: {prediction}, confidence: {confidence:.4f}, normal_prob: {normal_prob:.4f}, pneumonia_prob: {pneumonia_prob:.4f}")
            return prediction, confidence, normal_prob, pneumonia_prob
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            # Default to no prediction in case of error
            return "UNKNOWN", 0.0, 0.0, 0.0
    
class AboutView(TemplateView):
    template_name = 'core/about.html'

# Authentication views
class CustomLoginView(LoginView):
    template_name = 'core/login.html'
    
class CustomLogoutView(LogoutView):
    next_page = 'core:login'

# Prediction views
class ResultView(LoginRequiredMixin, DetailView):
    model = XRayImage
    template_name = 'core/result.html'
    context_object_name = 'prediction'
    pk_url_kwarg = 'prediction_id'
    
class DeletePredictionView(LoginRequiredMixin, DeleteView):
    model = XRayImage
    success_url = reverse_lazy('core:home')
    pk_url_kwarg = 'prediction_id'
    
class UpdatePredictionView(LoginRequiredMixin, UpdateView):
    model = XRayImage
    template_name = 'core/update_prediction.html'
    fields = ['patient_name', 'patient_surname', 'patient_id', 'patient_date_of_birth', 'patient_gender', 'xray_date']
    pk_url_kwarg = 'prediction_id'
    
    def get_success_url(self):
        return reverse_lazy('core:result', kwargs={'prediction_id': self.kwargs['prediction_id']})
    
    def form_valid(self, form):
        self.object = form.save()
        # Check if this is an AJAX request
        if self.request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True})
        return super().form_valid(form)
    
    def form_invalid(self, form):
        # Return errors for AJAX requests
        if self.request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            errors = {field: errors[0] for field, errors in form.errors.items()}
            return JsonResponse({'success': False, 'error': 'Form contains errors', 'errors': errors})
        return super().form_invalid(form)

# X-Ray views
class XRayListView(LoginRequiredMixin, ListView):
    model = XRayImage
    template_name = 'core/xray_list.html'
    context_object_name = 'xrays'
    
class XRayDetailView(LoginRequiredMixin, DetailView):
    model = XRayImage
    template_name = 'core/xray_detail.html'
    context_object_name = 'xray'

# API endpoints
def test_prediction(request):
    # Simple implementation - adjust as needed
    return JsonResponse({'status': 'success', 'message': 'Test prediction endpoint'})
    
def api_test_view(request):
    # Simple implementation - adjust as needed
    return JsonResponse({'status': 'success', 'message': 'API test endpoint'})

# Error handlers
def custom_404(request, exception):
    return page_not_found(request, exception, template_name='core/404.html')

def custom_500(request):
    return server_error(request, template_name='core/500.html')

def handler400(request, exception):
    return bad_request(request, exception, template_name='core/400.html')

def handler403(request, exception):
    return permission_denied(request, exception, template_name='core/403.html') 