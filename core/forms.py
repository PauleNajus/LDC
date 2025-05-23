from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from .models import XRayImage
import datetime

class XRayImageForm(forms.ModelForm):
    # Define custom gender choices
    GENDER_CHOICES = [
        ('', _('Select Gender')),
        ('M', _('Male')),
        ('F', _('Female')),
    ]

    # Override the gender field to use our custom choices
    patient_gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

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
    
    def clean_patient_gender(self):
        gender = self.cleaned_data.get('patient_gender')
        if gender:
            # Convert display values to model values
            gender_map = {
                'Male': 'M',
                'Female': 'F'
            }
            if gender in gender_map:
                return gender_map[gender]
            elif gender in ['M', 'F']:
                return gender
            else:
                raise ValidationError(_('Invalid gender selection'))
        return gender
    
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
        
        # Make only image field required, others optional
        for field_name, field in self.fields.items():
            if field_name == 'image':
                field.required = True
            else:
                field.required = False 

class PredictionSearchForm(forms.Form):
    search_query = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': _('Search by name, surname, or ID')
        })
    )
    date_from = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control date-input',
            'placeholder': _('From date (YYYY-MM-DD)'),
            'oninput': 'formatDate(this)',
            'onkeypress': 'return isNumberOrDash(event)'
        })
    )
    date_to = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control date-input',
            'placeholder': _('To date (YYYY-MM-DD)'),
            'oninput': 'formatDate(this)',
            'onkeypress': 'return isNumberOrDash(event)'
        })
    )
    prediction_type = forms.ChoiceField(
        required=False,
        choices=[
            ('', _('All')),
            ('NORMAL', _('Normal')),
            ('PNEUMONIA', _('Pneumonia'))
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    def clean_date_from(self):
        date_from = self.cleaned_data.get('date_from')
        if date_from:
            try:
                return datetime.datetime.strptime(date_from, '%Y-%m-%d').date()
            except ValueError:
                raise ValidationError(_('Invalid date format. Use YYYY-MM-DD'))
        return None

    def clean_date_to(self):
        date_to = self.cleaned_data.get('date_to')
        if date_to:
            try:
                return datetime.datetime.strptime(date_to, '%Y-%m-%d').date()
            except ValueError:
                raise ValidationError(_('Invalid date format. Use YYYY-MM-DD'))
        return None

    def clean(self):
        cleaned_data = super().clean()
        date_from = cleaned_data.get('date_from')
        date_to = cleaned_data.get('date_to')

        if date_from and date_to and date_from > date_to:
            raise ValidationError(_('From date cannot be later than To date'))

        return cleaned_data 