"""Gunicorn configuration file."""

import multiprocessing  # noqa

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:80"

timeout = 180  # 3 minutes

worker_class = "uvicorn.workers.UvicornWorker"
# workers = (multiprocessing.cpu_count() * 2) + 1
workers = 1
