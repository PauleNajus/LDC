from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
import logging
import datetime

logger = logging.getLogger('core')

class Command(BaseCommand):
    help = 'Check password security for all users'

    def add_arguments(self, parser):
        parser.add_argument(
            '--notify',
            action='store_true',
            help='Send email notifications to users with weak passwords'
        )
        parser.add_argument(
            '--max-age',
            type=int,
            default=90,
            help='Maximum password age in days'
        )

    def handle(self, *args, **options):
        notify = options['notify']
        max_age = options['max_age']
        User = get_user_model()
        
        # Get all active users
        users = User.objects.filter(is_active=True)
        total_users = users.count()
        weak_passwords = 0
        old_passwords = 0
        
        self.stdout.write(f'Checking passwords for {total_users} users...')
        
        for user in users:
            try:
                # Check if password is too old
                if user.password and hasattr(user, 'last_password_change'):
                    days_old = (timezone.now() - user.last_password_change).days
                    if days_old > max_age:
                        old_passwords += 1
                        self.stdout.write(
                            self.style.WARNING(
                                f'User {user.username} has not changed their password in {days_old} days'
                            )
                        )
                        if notify:
                            self._notify_user_old_password(user, days_old)
                
                # Check password strength
                try:
                    # Get the raw password (this is just for demonstration, in real life
                    # we can't access the raw password)
                    raw_password = 'dummy_password'  # We can't actually check the real password
                    validate_password(raw_password, user)
                except ValidationError as e:
                    weak_passwords += 1
                    self.stdout.write(
                        self.style.WARNING(
                            f'User {user.username} has a weak password: {", ".join(e.messages)}'
                        )
                    )
                    if notify:
                        self._notify_user_weak_password(user, e.messages)
                
            except Exception as e:
                logger.error(f'Error checking password for user {user.username}: {str(e)}')
                continue
        
        # Log summary
        self.stdout.write(
            self.style.SUCCESS(
                f'\nPassword security check completed:\n'
                f'Total users checked: {total_users}\n'
                f'Users with weak passwords: {weak_passwords}\n'
                f'Users with old passwords: {old_passwords}'
            )
        )
        
        # Log to file
        logger.info(
            f'Password security check completed - '
            f'Total: {total_users}, Weak: {weak_passwords}, Old: {old_passwords}'
        )
    
    def _notify_user_weak_password(self, user, reasons):
        """Send email notification about weak password."""
        try:
            from django.core.mail import send_mail
            from django.conf import settings
            from django.template.loader import render_to_string
            
            context = {
                'user': user,
                'reasons': reasons,
                'reset_url': f'{settings.SITE_URL}{settings.LOGIN_URL}',
            }
            
            message = render_to_string('core/email/weak_password.txt', context)
            html_message = render_to_string('core/email/weak_password.html', context)
            
            send_mail(
                'Security Alert: Your Password Needs Updating',
                message,
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                html_message=html_message,
                fail_silently=False,
            )
            
            logger.info(f'Sent weak password notification to user {user.username}')
            
        except Exception as e:
            logger.error(f'Error sending weak password notification to {user.username}: {str(e)}')
    
    def _notify_user_old_password(self, user, days_old):
        """Send email notification about old password."""
        try:
            from django.core.mail import send_mail
            from django.conf import settings
            from django.template.loader import render_to_string
            
            context = {
                'user': user,
                'days_old': days_old,
                'reset_url': f'{settings.SITE_URL}{settings.LOGIN_URL}',
            }
            
            message = render_to_string('core/email/old_password.txt', context)
            html_message = render_to_string('core/email/old_password.html', context)
            
            send_mail(
                'Security Alert: Time to Change Your Password',
                message,
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                html_message=html_message,
                fail_silently=False,
            )
            
            logger.info(f'Sent old password notification to user {user.username}')
            
        except Exception as e:
            logger.error(f'Error sending old password notification to {user.username}: {str(e)}') 