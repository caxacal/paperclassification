# gunicorn.conf.py

# SciBERT-safe config for Render Standard
workers = 1
worker_class = "sync"

timeout = 180
graceful_timeout = 180

preload_app = True

# Prevent memory leaks
max_requests = 100
max_requests_jitter = 20
