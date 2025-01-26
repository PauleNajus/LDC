from django import template
from django.template.defaultfilters import floatformat
from django.utils.safestring import mark_safe
import datetime
import os
from django.utils.translation import gettext as _

register = template.Library()

@register.filter
def confidence_color(value):
    """Return a color class based on confidence value."""
    try:
        value = float(value)
        if value >= 80:
            return 'text-green-600'
        elif value >= 60:
            return 'text-yellow-600'
        else:
            return 'text-red-600'
    except (ValueError, TypeError):
        return ''

@register.filter
def format_percentage(value):
    """Format a number as a percentage with 2 decimal places."""
    try:
        value = float(value)
        return f"{floatformat(value, 2)}%"
    except (ValueError, TypeError):
        return '0.00%'

@register.filter
def format_time(value):
    """Format processing time in milliseconds."""
    try:
        value = float(value)
        if value < 0.001:
            return f"{value * 1000000:.2f} μs"
        elif value < 1:
            return f"{value * 1000:.2f} ms"
        else:
            return f"{value:.2f} s"
    except (ValueError, TypeError):
        return '0.00 s'

@register.filter
def format_file_size(value):
    """Format file size in bytes to human readable format."""
    try:
        value = float(value)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if value < 1024:
                return f"{value:.2f} {unit}"
            value /= 1024
        return f"{value:.2f} TB"
    except (ValueError, TypeError):
        return '0 B'

@register.filter
def format_date(value):
    """Format date string to a more readable format."""
    if not value or value == "No data":
        return "No data"
    try:
        if isinstance(value, str):
            date = datetime.datetime.strptime(value, '%Y-%m-%d')
        else:
            date = value
        return date.strftime('%B %d, %Y')
    except (ValueError, TypeError):
        return value

@register.simple_tag
def prediction_badge(prediction, confidence):
    """Return a colored badge for prediction result."""
    try:
        confidence = float(confidence)
        if prediction == 'NORMAL':
            color_class = 'bg-green-100 text-green-800'
            if confidence < 60:
                color_class = 'bg-yellow-100 text-yellow-800'
        else:  # PNEUMONIA
            color_class = 'bg-red-100 text-red-800'
            if confidence < 60:
                color_class = 'bg-yellow-100 text-yellow-800'
        
        return mark_safe(f'<span class="px-2 py-1 rounded-full text-sm font-semibold {color_class}">{prediction}</span>')
    except (ValueError, TypeError):
        return prediction

@register.simple_tag
def progress_bar(value, max_value=100):
    """Return a progress bar with the given value."""
    try:
        percentage = min(100, max(0, float(value)))
        if percentage >= 80:
            color_class = 'bg-green-600'
        elif percentage >= 60:
            color_class = 'bg-yellow-600'
        else:
            color_class = 'bg-red-600'
        
        return mark_safe(f'''
            <div class="w-full bg-gray-200 rounded-full h-2.5">
                <div class="{color_class} h-2.5 rounded-full" style="width: {percentage}%"></div>
            </div>
        ''')
    except (ValueError, TypeError):
        return ''

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary using bracket notation in templates."""
    return dictionary.get(key)

@register.filter
def mask_patient_id(value):
    """Mask part of the patient ID for privacy."""
    if not value or value == "No data":
        return value
    try:
        if len(value) <= 4:
            return '*' * len(value)
        return value[:2] + '*' * (len(value) - 4) + value[-2:]
    except (ValueError, TypeError):
        return value

@register.simple_tag
def gender_icon(gender):
    """Return a gender icon based on the gender value."""
    icons = {
        'M': '<i class="fas fa-mars text-blue-600"></i>',
        'F': '<i class="fas fa-venus text-pink-600"></i>',
        'O': '<i class="fas fa-transgender text-purple-600"></i>',
        'N': '<i class="fas fa-user text-gray-600"></i>',
    }
    return mark_safe(icons.get(gender, icons['N']))

@register.filter
def age_from_dob(dob):
    """Calculate age from date of birth."""
    if not dob or dob == "No data":
        return "Unknown"
    try:
        if isinstance(dob, str):
            birth_date = datetime.datetime.strptime(dob, '%Y-%m-%d').date()
        else:
            birth_date = dob
        today = datetime.date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return f"{age} years"
    except (ValueError, TypeError):
        return "Unknown"

@register.filter
def filename(value):
    return os.path.basename(str(value)) if value else _("No filename") 