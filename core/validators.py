from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.conf import settings
from django.core.validators import RegexValidator
import re

class ComplexPasswordValidator:
    """
    Validate that the password meets complexity requirements:
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    - No common patterns
    """
    
    def __init__(self):
        # Get patterns from settings or use defaults
        self.password_patterns = getattr(settings, 'COMPLEX_PASSWORD_PATTERNS', [
            r'password',
            r'123',
            r'abc',
            r'qwerty',
            r'admin',
            r'letmein',
            r'welcome',
        ])
        
        # Create regex validators
        self.validators = [
            RegexValidator(
                regex=r'[A-Z]',
                message=_("Password must contain at least one uppercase letter."),
                code='password_no_upper',
                inverse_match=True
            ),
            RegexValidator(
                regex=r'[a-z]',
                message=_("Password must contain at least one lowercase letter."),
                code='password_no_lower',
                inverse_match=True
            ),
            RegexValidator(
                regex=r'[0-9]',
                message=_("Password must contain at least one digit."),
                code='password_no_digit',
                inverse_match=True
            ),
            RegexValidator(
                regex=r'[!@#$%^&*(),.?":{}|<>]',
                message=_("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)."),
                code='password_no_special',
                inverse_match=True
            ),
        ]
    
    def validate(self, password, user=None):
        # Run all regex validators
        for validator in self.validators:
            try:
                validator(password)
            except ValidationError as e:
                raise ValidationError(e.message, code=e.code)
        
        # Check for common patterns
        password_lower = password.lower()
        for pattern in self.password_patterns:
            if pattern in password_lower:
                raise ValidationError(
                    _("Password contains a common pattern that is too easy to guess."),
                    code='password_common_pattern',
                )
    
    def get_help_text(self):
        return _(
            "Your password must contain at least: one uppercase letter, "
            "one lowercase letter, one digit, and one special character. "
            "It cannot contain common patterns like '123' or 'password'."
        )

class NoPersonalInfoValidator:
    """
    Validate that the password doesn't contain personal information.
    Uses Django's built-in user attribute handling.
    """
    
    def __init__(self):
        self.user_attributes = getattr(settings, 'PERSONAL_INFO_ATTRIBUTES', [
            'username',
            'first_name',
            'last_name',
            'email',
        ])
        self.min_length = getattr(settings, 'PERSONAL_INFO_MIN_LENGTH', 3)
    
    def validate(self, password, user=None):
        if not user:
            return
        
        password_lower = password.lower()
        
        # Check each configured user attribute
        for attribute in self.user_attributes:
            value = getattr(user, attribute, None)
            if not value:
                continue
                
            if attribute == 'email':
                # Split email and check each part
                local_part = value.split('@')[0]
                parts = local_part.split('.') + [local_part]
                for part in parts:
                    if len(part) >= self.min_length and part.lower() in password_lower:
                        raise ValidationError(
                            _("Password cannot contain parts of your email address."),
                            code='password_email_info'
                        )
            else:
                # Check other attributes
                if len(value) >= self.min_length and value.lower() in password_lower:
                    raise ValidationError(
                        _("Password cannot contain your %(attribute)s.") % {'attribute': attribute},
                        code=f'password_{attribute}_info',
                        params={'attribute': attribute}
                    )
    
    def get_help_text(self):
        return _(
            "Your password cannot contain your personal information "
            "(such as username, name, or email address)."
        )

class NoRepeatedCharactersValidator:
    """
    Validate that the password doesn't contain too many repeated characters
    or keyboard patterns.
    """
    
    def __init__(self, max_repeats=None):
        self.max_repeats = max_repeats or getattr(settings, 'MAX_REPEATED_CHARS', 3)
        self.keyboard_layouts = getattr(settings, 'KEYBOARD_LAYOUTS', {
            'qwerty': [
                'qwertyuiop',
                'asdfghjkl',
                'zxcvbnm'
            ],
            'azerty': [
                'azertyuiop',
                'qsdfghjklm',
                'wxcvbn'
            ]
        })
        self.min_pattern_length = getattr(settings, 'MIN_KEYBOARD_PATTERN_LENGTH', 4)
    
    def validate(self, password, user=None):
        # Check for repeated characters
        last_char = None
        repeat_count = 1
        
        for char in password:
            if char == last_char:
                repeat_count += 1
                if repeat_count > self.max_repeats:
                    raise ValidationError(
                        _("Password cannot contain more than %(max_repeats)d repeated characters.") % {
                            'max_repeats': self.max_repeats
                        },
                        code='password_repeated_chars',
                        params={'max_repeats': self.max_repeats}
                    )
            else:
                repeat_count = 1
                last_char = char
        
        # Check for keyboard patterns in all layouts
        password_lower = password.lower()
        for layout_name, keyboard_rows in self.keyboard_layouts.items():
            # Forward patterns
            for row in keyboard_rows:
                for i in range(len(row) - self.min_pattern_length + 1):
                    pattern = row[i:i + self.min_pattern_length]
                    if pattern in password_lower:
                        raise ValidationError(
                            _("Password cannot contain keyboard pattern '%(pattern)s' from %(layout)s layout.") % {
                                'pattern': pattern,
                                'layout': layout_name.upper()
                            },
                            code='password_keyboard_pattern',
                            params={'pattern': pattern, 'layout': layout_name}
                        )
                    
                    # Check reverse pattern
                    pattern_reverse = pattern[::-1]
                    if pattern_reverse in password_lower:
                        raise ValidationError(
                            _("Password cannot contain reversed keyboard pattern '%(pattern)s' from %(layout)s layout.") % {
                                'pattern': pattern_reverse,
                                'layout': layout_name.upper()
                            },
                            code='password_keyboard_pattern_reverse',
                            params={'pattern': pattern_reverse, 'layout': layout_name}
                        )
    
    def get_help_text(self):
        return _(
            "Your password cannot contain more than %(max_repeats)d repeated characters "
            "or common keyboard patterns from standard layouts."
        ) % {'max_repeats': self.max_repeats} 