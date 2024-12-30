from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils import timezone
from .models import XRayImage
from .models import LungClassifier

def home(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:
            messages.error(request, 'Please select an image to upload.')
            return redirect('core:home')

        image = request.FILES['image']
        if not image.content_type.startswith('image/'):
            messages.error(request, 'Please upload a valid image file.')
            return redirect('core:home')

        try:
            # Create XRayImage instance
            xray = XRayImage.objects.create(
                image=image,
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

        except Exception as e:
            messages.error(request, f'Error processing image: {str(e)}')
            return redirect('core:home')

    recent_predictions = XRayImage.objects.order_by('-uploaded_at')[:6]
    return render(request, 'core/home.html', {'recent_predictions': recent_predictions})

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