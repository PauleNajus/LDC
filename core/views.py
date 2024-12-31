from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.core.exceptions import ValidationError
from .models import XRayImage
from .models import LungClassifier
from datetime import datetime

def format_date(date_str):
    """Format date to Lithuanian format or return 'No data' if empty"""
    if not date_str:
        return "No data"
    try:
        # Parse the date and format it to Lithuanian format
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return "No data"

@ensure_csrf_cookie
def home(request):
    try:
        if request.method == 'POST':
            if 'image' not in request.FILES:
                messages.error(request, 'Please select an image to upload.')
                return redirect('core:home')

            image = request.FILES['image']
            if not image.content_type.startswith('image/'):
                messages.error(request, 'Please upload a valid image file.')
                return redirect('core:home')

            try:
                # Validate dates
                birth_date = format_date(request.POST.get('patient_date_of_birth', ''))
                xray_date = format_date(request.POST.get('xray_date', ''))

                # Create XRayImage instance with patient information
                xray = XRayImage.objects.create(
                    image=image,
                    patient_name=request.POST.get('patient_name', 'No data'),
                    patient_surname=request.POST.get('patient_surname', 'No data'),
                    patient_id=request.POST.get('patient_id', 'No data'),
                    patient_date_of_birth=birth_date,
                    patient_gender=request.POST.get('patient_gender', 'No data'),
                    xray_date=xray_date,
                    uploaded_at=timezone.now()
                )

                # Initialize classifier and make prediction
                classifier = LungClassifier()
                result = classifier.predict(xray.image.path)

                # Update XRayImage with prediction results
                xray.prediction = result['class']
                xray.confidence = result['confidence']
                xray.normal_probability = result['normal_probability']
                xray.pneumonia_probability = result['pneumonia_probability']
                xray.processing_time = result['processing_time']
                xray.image_size = result['image_size']
                xray.save()

                messages.success(request, 'Image uploaded and analyzed successfully!')
                return redirect('core:home')

            except ValidationError as e:
                messages.error(request, f'Invalid data provided: {str(e)}')
                return redirect('core:home')
            except Exception as e:
                messages.error(request, f'Error processing image: {str(e)}')
                return redirect('core:home')

        recent_predictions = XRayImage.objects.order_by('-uploaded_at')[:6]
        return render(request, 'core/home.html', {'recent_predictions': recent_predictions})
    
    except Exception as e:
        messages.error(request, f'An unexpected error occurred: {str(e)}')
        return render(request, 'core/home.html', {'recent_predictions': []})

def about(request):
    model_info = {
        'Architecture': 'Convolutional Neural Network (CNN)',
        'Framework': 'PyTorch 2.5.1',
        'Input Size': '224x224 pixels',
        'Classes': 'Normal, Pneumonia',
        'Training Dataset': 'Chest X-Ray Images',
        'GPU Acceleration': 'CUDA (if available)',
        'Model Size': '~25MB'
    }
    return render(request, 'core/about.html', {'model_info': model_info})

def result(request, prediction_id):
    xray = get_object_or_404(XRayImage, id=prediction_id)
    context = {
        'image_url': xray.image.url,
        'prediction': xray.prediction,
        'confidence': xray.confidence,
        'timestamp': xray.uploaded_at,
    }
    return render(request, 'core/result.html', context)

def delete_prediction(request, prediction_id):
    try:
        xray = get_object_or_404(XRayImage, id=prediction_id)
        xray.image.delete()  # Delete the image file
        xray.delete()  # Delete the database record
        messages.success(request, 'Record deleted successfully.')
    except Exception as e:
        messages.error(request, f'Error deleting record: {str(e)}')
    return redirect('core:home')