# Generated by Django 5.1.4 on 2025-01-04 18:28

import core.models
import django.core.validators
import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        ('core', '0005_alter_xrayimage_image_size_and_more'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='user',
            options={'ordering': ['-date_joined'], 'verbose_name': 'User', 'verbose_name_plural': 'Users'},
        ),
        migrations.AlterModelOptions(
            name='xrayimage',
            options={'ordering': ['-uploaded_at'], 'verbose_name': 'X-Ray Image', 'verbose_name_plural': 'X-Ray Images'},
        ),
        migrations.AddField(
            model_name='user',
            name='failed_login_attempts',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='user',
            name='last_failed_login',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='user',
            name='last_password_change',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='user',
            name='password_history',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='user',
            name='security_questions',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='xrayimage',
            name='user',
            field=models.ForeignKey(default=core.models.get_default_user, on_delete=django.db.models.deletion.SET_DEFAULT, related_name='xray_images', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='user',
            name='first_name',
            field=models.CharField(max_length=30, validators=[django.core.validators.MinLengthValidator(2)]),
        ),
        migrations.AlterField(
            model_name='user',
            name='last_name',
            field=models.CharField(max_length=30, validators=[django.core.validators.MinLengthValidator(2)]),
        ),
        migrations.AlterField(
            model_name='xrayimage',
            name='image',
            field=models.ImageField(upload_to='xray_images/', validators=[core.models.validate_image_extension, core.models.validate_image_size]),
        ),
        migrations.AlterField(
            model_name='xrayimage',
            name='patient_gender',
            field=models.CharField(choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other'), ('N', 'Not Specified')], default='N', max_length=1),
        ),
        migrations.AlterField(
            model_name='xrayimage',
            name='patient_id',
            field=models.CharField(default='No data', max_length=50, validators=[django.core.validators.RegexValidator(message='Patient ID can only contain letters, numbers, and hyphens', regex='^[A-Za-z0-9-]+$')]),
        ),
        migrations.AlterField(
            model_name='xrayimage',
            name='patient_name',
            field=models.CharField(default='No data', max_length=100, validators=[django.core.validators.MinLengthValidator(2)]),
        ),
        migrations.AlterField(
            model_name='xrayimage',
            name='patient_surname',
            field=models.CharField(default='No data', max_length=100, validators=[django.core.validators.MinLengthValidator(2)]),
        ),
        migrations.AlterField(
            model_name='xrayimage',
            name='uploaded_at',
            field=models.DateTimeField(auto_now_add=True, db_index=True),
        ),
        migrations.AddIndex(
            model_name='user',
            index=models.Index(fields=['email'], name='core_user_email_38052c_idx'),
        ),
        migrations.AddIndex(
            model_name='user',
            index=models.Index(fields=['username'], name='core_user_usernam_e8adca_idx'),
        ),
        migrations.AddIndex(
            model_name='user',
            index=models.Index(fields=['date_joined'], name='core_user_date_jo_a935f6_idx'),
        ),
        migrations.AddIndex(
            model_name='user',
            index=models.Index(fields=['last_password_change'], name='core_user_last_pa_9ed543_idx'),
        ),
        migrations.AddIndex(
            model_name='xrayimage',
            index=models.Index(fields=['prediction'], name='core_xrayim_predict_9d4f24_idx'),
        ),
        migrations.AddIndex(
            model_name='xrayimage',
            index=models.Index(fields=['patient_id'], name='core_xrayim_patient_b9d64d_idx'),
        ),
        migrations.AddIndex(
            model_name='xrayimage',
            index=models.Index(fields=['uploaded_at'], name='core_xrayim_uploade_dca8bb_idx'),
        ),
    ]
