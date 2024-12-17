from prometheus_client import Counter, Histogram, Gauge, start_http_server
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class PrometheusConfig:
    port: int = 8001
    enabled: bool = False  # Default to disabled
    path: str = "/metrics"
    startup_server: bool = False  # Whether to start the HTTP server internally

class PrometheusMetrics:
    def __init__(self, config: Optional[PrometheusConfig] = None):
        # Allow environment variables to override config
        self.config = config or PrometheusConfig()
        
        # Environment overrides
        if os.getenv('ENABLE_METRICS'):
            self.config.enabled = os.getenv('ENABLE_METRICS').lower() == 'true'
        if os.getenv('METRICS_PORT'):
            self.config.port = int(os.getenv('METRICS_PORT'))
        if os.getenv('METRICS_PATH'):
            self.config.path = os.getenv('METRICS_PATH')
            
        if not self.config.enabled:
            return

        # Initialize metrics collectors
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

    def record_request(self, endpoint: str):
        if not self.config.enabled:
            return
        self.requests_total.labels(endpoint=endpoint).inc()

    def record_latency(self, endpoint: str, duration: float):
        if not self.config.enabled:
            return
        self.request_duration.labels(endpoint=endpoint).observe(duration)

    def record_tokens(self, input_count: int, output_count: int):
        if not self.config.enabled:
            return
        self.token_throughput.labels(direction="input").inc(input_count)
        self.token_throughput.labels(direction="output").inc(output_count)

    def record_error(self, error_type: str):
        if not self.config.enabled:
            return
        self.errors_total.labels(error_type=error_type).inc()

    def update_gpu_memory(self, device: str, bytes_used: int):
        if not self.config.enabled:
            return
        self.gpu_memory_usage.labels(device=device).set(bytes_used)

    @classmethod
    def create_standalone(cls, port: int = 8001) -> 'PrometheusMetrics':
        """Create a standalone metrics server instance"""
        return cls(PrometheusConfig(
            port=port,
            enabled=True,
            startup_server=True
        ))