import logging
import time
import psutil
import os
from typing import Any, Callable
from django.http import HttpRequest, HttpResponse
from django.conf import settings
from django.shortcuts import redirect
from django.urls import resolve, reverse

logger = logging.getLogger('django')

class RequestLoggingMiddleware:
    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Start timer and get initial system stats
        start_time: float = time.time()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        initial_cpu = psutil.cpu_percent()
        
        # Log request details
        if request.method == 'POST' and request.FILES:
            logger.info('\n' + '='*50)
            logger.info('=== Incoming File Upload Request ===')
            logger.info(f'Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            logger.info(f'Request Path: {request.path}')
            logger.info(f'Content Type: {request.content_type}')
            logger.info(f'User Agent: {request.META.get("HTTP_USER_AGENT", "Unknown")}')
            logger.info(f'Client IP: {request.META.get("REMOTE_ADDR", "Unknown")}')
            
            # Log system state before processing
            logger.info('\n=== System State Before Processing ===')
            logger.info(f'CPU Usage: {initial_cpu:.1f}%')
            logger.info(f'Memory Usage: {initial_memory:.1f} MB')
            logger.info(f'System Memory: {psutil.virtual_memory().percent}% used')
            
            # Log file details
            logger.info('\n=== File Upload Details ===')
            for filename, file in request.FILES.items():
                logger.info(f'File: {filename}')
                logger.info(f' - Size: {file.size / 1024:.2f} KB')
                logger.info(f' - Content Type: {file.content_type}')
                logger.info(f' - Charset: {getattr(file, "charset", "Unknown")}')
        
        # Process request
        response = self.get_response(request)
        
        # Calculate final stats
        end_time: float = time.time()
        duration: float = end_time - start_time
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        final_cpu = psutil.cpu_percent()
        
        # Log response details for file uploads
        if request.method == 'POST' and request.FILES:
            logger.info('\n=== Response Details ===')
            logger.info(f'Status Code: {response.status_code}')
            logger.info(f'Content Type: {response.get("Content-Type", "Unknown")}')
            logger.info(f'Processing Time: {duration:.3f} seconds')
            
            # Log system state after processing
            logger.info('\n=== System State After Processing ===')
            logger.info(f'CPU Usage: {final_cpu:.1f}%')
            logger.info(f'Memory Usage: {final_memory:.1f} MB')
            logger.info(f'Memory Change: {final_memory - initial_memory:+.1f} MB')
            logger.info(f'System Memory: {psutil.virtual_memory().percent}% used')
            
            # Log performance summary
            logger.info('\n=== Performance Summary ===')
            logger.info(f'Total Processing Time: {duration:.3f} seconds')
            logger.info(f'CPU Usage - Initial: {initial_cpu:.1f}%, Final: {final_cpu:.1f}%')
            logger.info(f'Memory Impact: {final_memory - initial_memory:+.1f} MB')
            logger.info('='*50 + '\n')
        
        return response

class SecurityMiddleware:
    def __init__(self, get_response: Callable):
        self.get_response = get_response
        self.logger = logging.getLogger('django.security')

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)
        return response

class AuthenticationMiddleware:
    def __init__(self, get_response: Callable):
        self.get_response = get_response
        self.logger = logging.getLogger('django.auth')
        # Paths that don't require authentication
        self.public_paths = [
            reverse('core:login'),
            '/admin/login/',
            '/static/',
            '/media/',
        ]

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Skip middleware for paths that are public
        path = request.path
        if any(path.startswith(public_path) for public_path in self.public_paths):
            return self.get_response(request)
        
        # Check if user is authenticated
        if not request.user.is_authenticated:
            # Redirect to login page
            login_url = reverse('core:login')
            return redirect(login_url)
        
        return self.get_response(request) 