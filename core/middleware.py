import logging
import time
import json
import psutil
import threading
from django.utils.deprecation import MiddlewareMixin
from django.core.cache import cache
from django.conf import settings
from django.http import JsonResponse
from django.utils.translation import activate
from django.utils import translation
from django.urls import is_valid_path
from django.http import HttpResponseRedirect
from django.urls.base import get_script_prefix

logger = logging.getLogger('core')

class RequestMonitor:
    """Monitor system resources during request processing."""
    
    def __init__(self):
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent()
        self.start_memory = psutil.virtual_memory().percent
        
    def get_stats(self):
        """Get resource usage statistics."""
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        
        return {
            'duration_ms': round((end_time - self.start_time) * 1000, 2),
            'cpu_usage': round(end_cpu - self.start_cpu, 2),
            'memory_usage': round(end_memory - self.start_memory, 2),
        }

class RequestLoggingMiddleware(MiddlewareMixin):
    """Middleware to log all requests and responses with resource monitoring."""
    
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self._requests = {}
        self._lock = threading.Lock()
    
    def process_request(self, request):
        """Set the request start time and initialize monitoring."""
        request.id = id(request)
        with self._lock:
            self._requests[request.id] = RequestMonitor()

    def process_response(self, request, response):
        """Log the request and response details with resource usage."""
        try:
            # Get monitoring stats
            with self._lock:
                monitor = self._requests.pop(request.id, None)
            
            if monitor:
                stats = monitor.get_stats()
            else:
                stats = {'duration_ms': 0, 'cpu_usage': 0, 'memory_usage': 0}
            
            # Get basic request details
            log_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'method': request.method,
                'path': request.path,
                'status_code': response.status_code,
                'user': str(request.user) if request.user.is_authenticated else 'anonymous',
                'ip': self.get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                'referer': request.META.get('HTTP_REFERER', ''),
                'content_type': response.get('Content-Type', ''),
                'content_length': len(response.content) if hasattr(response, 'content') else 0,
                'performance': stats
            }
            
            # Log request parameters if any
            if request.method in ['POST', 'PUT', 'PATCH']:
                if request.content_type and 'application/json' in request.content_type:
                    try:
                        log_data['body'] = json.loads(request.body)
                    except json.JSONDecodeError:
                        log_data['body'] = 'Invalid JSON body'
                else:
                    log_data['body'] = dict(request.POST)
                    
                # Remove sensitive information
                self.sanitize_data(log_data['body'])
            
            # Add response details for non-success status codes
            if response.status_code >= 400:
                try:
                    response_content = response.content.decode('utf-8')
                    if len(response_content) > 1000:
                        response_content = response_content[:1000] + '...'
                    log_data['response'] = response_content
                except Exception:
                    log_data['response'] = 'Unable to decode response content'
            
            # Log slow requests
            if stats['duration_ms'] > 1000:  # Requests taking more than 1 second
                logger.warning(f"Slow request detected: {json.dumps(log_data, default=str)}")
            
            # Log high resource usage
            if stats['cpu_usage'] > 50 or stats['memory_usage'] > 50:
                logger.warning(f"High resource usage detected: {json.dumps(log_data, default=str)}")
            
            # Log all requests at info level
            logger.info(f"Request processed: {json.dumps(log_data, default=str)}")
            
            # Update request statistics in cache
            self.update_request_stats(log_data)
            
        except Exception as e:
            logger.error(f"Error in request logging middleware: {str(e)}")
        
        return response
    
    def get_client_ip(self, request):
        """Get the client's IP address from the request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR')
    
    def sanitize_data(self, data):
        """Remove sensitive information from the data."""
        sensitive_fields = ['password', 'token', 'key', 'secret', 'credit_card']
        if isinstance(data, dict):
            for key in list(data.keys()):
                if any(field in key.lower() for field in sensitive_fields):
                    data[key] = '[REDACTED]'
                elif isinstance(data[key], (dict, list)):
                    self.sanitize_data(data[key])
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self.sanitize_data(item)
    
    def update_request_stats(self, log_data):
        """Update request statistics in cache."""
        try:
            # Get current stats from cache
            stats_key = 'request_stats'
            stats = cache.get(stats_key) or {
                'total_requests': 0,
                'status_codes': {},
                'methods': {},
                'paths': {},
                'response_times': {
                    'avg': 0,
                    'min': float('inf'),
                    'max': 0
                }
            }
            
            # Update statistics
            stats['total_requests'] += 1
            stats['status_codes'][str(log_data['status_code'])] = (
                stats['status_codes'].get(str(log_data['status_code']), 0) + 1
            )
            stats['methods'][log_data['method']] = (
                stats['methods'].get(log_data['method'], 0) + 1
            )
            stats['paths'][log_data['path']] = (
                stats['paths'].get(log_data['path'], 0) + 1
            )
            
            # Update response time statistics
            duration = log_data['performance']['duration_ms']
            stats['response_times']['min'] = min(stats['response_times']['min'], duration)
            stats['response_times']['max'] = max(stats['response_times']['max'], duration)
            
            # Calculate moving average
            old_avg = stats['response_times']['avg']
            stats['response_times']['avg'] = (
                (old_avg * (stats['total_requests'] - 1) + duration) / 
                stats['total_requests']
            )
            
            # Store updated stats in cache
            cache.set(stats_key, stats, timeout=3600)  # Cache for 1 hour
            
        except Exception as e:
            logger.error(f"Error updating request stats: {str(e)}")

class SecurityMiddleware(MiddlewareMixin):
    """Middleware to handle security headers and checks."""
    
    def process_response(self, request, response):
        """Add security headers to response."""
        # Security headers
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        if not settings.DEBUG:
            response['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response
    
    def process_request(self, request):
        """Perform security checks on request."""
        # Rate limiting check (if not already handled by django-ratelimit)
        if not hasattr(request, 'limited'):
            key = f"ratelimit_{self.get_client_ip(request)}"
            requests = cache.get(key, 0)
            
            if requests > 1000:  # More than 1000 requests per hour
                logger.warning(f"Rate limit exceeded for IP: {self.get_client_ip(request)}")
                response = JsonResponse({
                    'error': 'Too many requests',
                    'detail': 'Please try again later'
                }, status=429)
                return response
            
            cache.set(key, requests + 1, timeout=3600)  # 1 hour timeout
    
    def get_client_ip(self, request):
        """Get the client's IP address from the request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR') 

class LanguageMiddleware(MiddlewareMixin):
    """
    Middleware to handle language selection.
    Priority:
    1. Language from session
    2. Language from user profile (if authenticated)
    3. Language from browser
    4. Default language (English)
    """
    def process_request(self, request):
        language = None
        
        # Check session first
        language = request.session.get('django_language')
        
        # If no language in session and user is authenticated, check user profile
        if not language and request.user.is_authenticated:
            language = request.user.language_preference
            if language:
                request.session['django_language'] = language
        
        # If still no language, check browser settings
        if not language:
            accept_language = request.META.get('HTTP_ACCEPT_LANGUAGE', '')
            for lang in accept_language.split(','):
                lang = lang.split(';')[0].strip()
                if lang in [code for code, name in settings.LANGUAGES]:
                    language = lang
                    break
        
        # If no language is found, use default
        if not language:
            language = settings.LANGUAGE_CODE
            
        # Activate the language
        activate(language)
        request.LANGUAGE_CODE = language
        
    def process_response(self, request, response):
        # Save current language to session if it has changed
        if hasattr(request, 'LANGUAGE_CODE'):
            request.session['django_language'] = request.LANGUAGE_CODE
        return response 