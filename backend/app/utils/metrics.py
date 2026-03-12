from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("watchtower_requests_total", "Total API requests", ['endpoint', 'method'])
REQUEST_ERRORS = Counter("watchtower_errors_total", "Total API errors", ['endpoint'])
REQUEST_LATENCY = Histogram("watchtower_request_latency_seconds", "Request latency seconds", ['endpoint'])

MODEL_INFERENCE_LATENCY = Histogram(
    "watchtower_model_inference_seconds", 
    "Time spent only in model loading and prediction logic (excluding network overhead)", 
    ['model_sha']
)