import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000
timeout = 30
keepalive = 2

# Process naming
proc_name = 'lung_classifier'
pythonpath = '.'

# Logging
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = 'info'

# SSL (uncomment and modify for HTTPS)
# keyfile = 'ssl/private.key'
# certfile = 'ssl/certificate.crt'

# Environment variables
raw_env = [
    'DJANGO_DEBUG=False',
    'DJANGO_SETTINGS_MODULE=lung_classifier.settings',
]

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190 