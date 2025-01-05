from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from .models import XRayImage
import datetime

class XRayImageForm(forms.ModelForm):
    class Meta:
        model = XRayImage
        fields = [
            'image',
            'patient_name',
            'patient_surname',
            'patient_id',
            'patient_date_of_birth',
            'patient_gender',
            'xray_date'
        ]
        widgets = {
            'patient_date_of_birth': forms.DateInput(attrs={'type': 'date'}),
            'xray_date': forms.DateInput(attrs={'type': 'date'}),
            'patient_gender': forms.Select(choices=XRayImage.GENDER_CHOICES),
        }
        labels = {
            'patient_name': _('First Name'),
            'patient_surname': _('Last Name'),
            'patient_id': _('Patient ID'),
            'patient_date_of_birth': _('Date of Birth'),
            'patient_gender': _('Gender'),
            'xray_date': _('X-Ray Date'),
        }
        help_texts = {
            'patient_id': _('Enter a unique identifier for the patient'),
            'image': _('Upload a chest X-ray image (JPEG, PNG)'),
        }
    
    def clean_patient_date_of_birth(self):
        date = self.cleaned_data.get('patient_date_of_birth')
        if date:
            # Convert string to date if necessary
            if isinstance(date, str):
                try:
                    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
                except ValueError:
                    raise ValidationError(_('Invalid date format. Use YYYY-MM-DD'))
            
            # Check if date is not in the future
            if date > datetime.date.today():
                raise ValidationError(_('Date of birth cannot be in the future'))
            
            # Check if date is not too far in the past (e.g., 150 years)
            min_date = datetime.date.today() - datetime.timedelta(days=150*365)
            if date < min_date:
                raise ValidationError(_('Date of birth is too far in the past'))
        
        return date
    
    def clean_xray_date(self):
        date = self.cleaned_data.get('xray_date')
        if date:
            # Convert string to date if necessary
            if isinstance(date, str):
                try:
                    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
                except ValueError:
                    raise ValidationError(_('Invalid date format. Use YYYY-MM-DD'))
            
            # Check if date is not in the future
            if date > datetime.date.today():
                raise ValidationError(_('X-ray date cannot be in the future'))
            
            # Check if date is not too far in the past (e.g., 10 years)
            min_date = datetime.date.today() - datetime.timedelta(days=10*365)
            if date < min_date:
                raise ValidationError(_('X-ray date is too old (maximum 10 years old)'))
            
            # Check if date is after date of birth
            dob = self.cleaned_data.get('patient_date_of_birth')
            if dob and isinstance(dob, datetime.date) and date < dob:
                raise ValidationError(_('X-ray date cannot be before date of birth'))
        
        return date
    
    def clean_patient_name(self):
        name = self.cleaned_data.get('patient_name')
        if name:
            # Remove extra whitespace and capitalize
            name = ' '.join(name.split()).title()
            
            # Check for valid characters
            if not all(c.isalpha() or c.isspace() for c in name):
                raise ValidationError(_('Name can only contain letters and spaces'))
            
            # Check minimum length
            if len(name) < 2:
                raise ValidationError(_('Name must be at least 2 characters long'))
        
        return name
    
    def clean_patient_surname(self):
        surname = self.cleaned_data.get('patient_surname')
        if surname:
            # Remove extra whitespace and capitalize
            surname = ' '.join(surname.split()).title()
            
            # Check for valid characters
            if not all(c.isalpha() or c.isspace() or c == '-' for c in surname):
                raise ValidationError(_('Surname can only contain letters, spaces, and hyphens'))
            
            # Check minimum length
            if len(surname) < 2:
                raise ValidationError(_('Surname must be at least 2 characters long'))
        
        return surname
    
    def clean_patient_id(self):
        patient_id = self.cleaned_data.get('patient_id')
        if patient_id:
            # Remove extra whitespace
            patient_id = patient_id.strip()
            
            # Check for valid characters
            if not all(c.isalnum() or c == '-' for c in patient_id):
                raise ValidationError(_('Patient ID can only contain letters, numbers, and hyphens'))
            
            # Check minimum length
            if len(patient_id) < 3:
                raise ValidationError(_('Patient ID must be at least 3 characters long'))
        
        return patient_id
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Check file size
            if image.size > 10 * 1024 * 1024:  # 10MB limit
                raise ValidationError(_('Image file size must be less than 10MB'))
            
            # Check file extension
            ext = image.name.split('.')[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png']:
                raise ValidationError(_('Only JPG and PNG files are allowed'))
            
            # Check image dimensions (if needed)
            from PIL import Image as PILImage
            img = PILImage.open(image)
            min_dimension = 200
            if img.width < min_dimension or img.height < min_dimension:
                raise ValidationError(_(f'Image dimensions must be at least {min_dimension}x{min_dimension} pixels'))
        
        return image
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add CSS classes for styling
        for field in self.fields:
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'placeholder': self.fields[field].label
            })
        
        # Make all fields required
        for field_name, field in self.fields.items():
            field.required = True 