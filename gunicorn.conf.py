# gunicorn.conf.py

workers = 1
worker_class = "sync"

timeout = 180
graceful_timeout = 180

preload_app = False   # 🔴 IMPORTANT: disable preload

max_requests = 100
max_requests_jitter = 20

def post_fork(server, worker):
    from app import cached_prediction
    try:
        cached_prediction("Warm up model for inference")
        server.log.info("Warm-up completed in worker")
    except Exception as e:
        server.log.warning("Warm-up failed", exc_info=e)
