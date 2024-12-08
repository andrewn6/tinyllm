from prometheus_client import Counter, Histogram, Gauge, start_http_server
from dataclasses import dataclass
from typing import Optional

@dataclass
class PrometheusConfig:
    port: int = 8001
    enabled: bool = True

class PrometheusMetrics:
    def __init__(self, config: Optional[PrometheusConfig] = None):
        self.config = config or PrometheusConfig()
        
        # Simple request metrics
        self.requests_total = Counter(
            'tinyllm_requests_total',
            'Total number of requests',
            ['endpoint']
        )
        
        self.request_duration = Histogram(
            'tinyllm_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        )
        
        # Token metrics
        self.token_throughput = Counter(
            'tinyllm_token_throughput_total',
            'Total tokens processed',
            ['direction']  # input/output
        )
        
        # Error tracking
        self.errors_total = Counter(
            'tinyllm_errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        # Resource metrics
        self.gpu_memory_usage = Gauge(
            'tinyllm_gpu_memory_bytes',
            'GPU memory usage in bytes',
            ['device']
        )

        if self.config.enabled:
            start_http_server(self.config.port)

    def record_request(self, endpoint: str):
        self.requests_total.labels(endpoint=endpoint).inc()

    def record_latency(self, endpoint: str, duration: float):
        self.request_duration.labels(endpoint=endpoint).observe(duration)

    def record_tokens(self, input_count: int, output_count: int):
        self.token_throughput.labels(direction="input").inc(input_count)
        self.token_throughput.labels(direction="output").inc(output_count)

    def record_error(self, error_type: str):
        self.errors_total.labels(error_type=error_type).inc()

    def update_gpu_memory(self, device: str, bytes_used: int):
        self.gpu_memory_usage.labels(device=device).set(bytes_used)