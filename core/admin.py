from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from .models import User, XRayImage

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'date_joined', 'last_login')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'date_joined')
    search_fields = ('username', 'first_name', 'last_name', 'email')
    ordering = ('-date_joined',)
    
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'email')}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
        }),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2'),
        }),
    )

@admin.register(XRayImage)
class XRayImageAdmin(admin.ModelAdmin):
    list_display = ('patient_full_name', 'patient_id', 'prediction', 'confidence_display', 'image_preview', 'uploaded_at')
    list_filter = ('prediction', 'uploaded_at', 'patient_gender')
    search_fields = ('patient_name', 'patient_surname', 'patient_id', 'prediction')
    readonly_fields = ('image_preview', 'confidence_display', 'uploaded_at', 'processing_time', 'image_size')
    ordering = ('-uploaded_at',)
    
    fieldsets = (
        ('Patient Information', {
            'fields': (
                'patient_name', 'patient_surname', 'patient_id',
                'patient_date_of_birth', 'patient_gender', 'xray_date'
            )
        }),
        ('Image Information', {
            'fields': ('image', 'image_preview', 'image_size')
        }),
        ('Prediction Results', {
            'fields': (
                'prediction', 'confidence_display',
                'normal_probability', 'pneumonia_probability',
                'processing_time'
            )
        }),
        ('Metadata', {
            'fields': ('uploaded_at', 'user')
        }),
    )
    
    def patient_full_name(self, obj):
        return f"{obj.patient_name} {obj.patient_surname}"
    patient_full_name.short_description = 'Patient Name'
    
    def confidence_display(self, obj):
        if obj.confidence > 80:
            color = 'green'
        elif obj.confidence > 60:
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {};">{:.2f}%</span>', color, obj.confidence)
    confidence_display.short_description = 'Confidence'
    
    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" style="max-width: 200px; max-height: 200px;" />', obj.image.url)
        return "No image"
    image_preview.short_description = 'Image Preview'
    
    def save_model(self, request, obj, form, change):
        if not change:  # If this is a new object
            obj.user = request.user
        super().save_model(request, obj, form, change) 