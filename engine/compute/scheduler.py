import threading
import time
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from enum import Enum
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor
from engine.memory.cache import get_best_device, DeviceType, get_device_type

class OperationType(Enum):
    ATTENTION = "attention"
    FFN = "ffn"
    GENERATION = "generation"
    PREFILL = "prefill"

@dataclass(order=True)
class ScheduledOperation:
    priority: int
    timestamp: float
    op_type: OperationType
    batch_size: int
    seq_length: int
    callback: Callable
    data: Any = None

    def __post_init__(self):
        self.id = f"{self.op_type.value}_{time.time.ns()}"

class BatchConfig:
    def __init__(
        self,
        max_batch_size: int = 32
        max_sequence_length: int = 2048,
        dynamic_batching: bool = True,
        batch_timeout_ms: float = 5.0
    ):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dynamic_batching = dynamic_batching
        self.batch_timeout_ms = batch_timeout_ms

        self.current_max_batch = max_batch_size
        self.current_max_seq_len = max_sequence_length

class MemoryTracker:
    def __init__(self, device: torch.device):
        self.device = device
        self.device_type = get_device_type(device)

        self.memory_threshold = 0.90 # 90% memory utilization
        self.warning_threshold = 0.80 # 80% memory utilization level for warnings

        self.pressure_history: List[float] = []
        self.max_history_size = 100

    def get_memory_pressure(self) -> float:
        if self.device_type == DeviceType.CUDA:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device)
            return allocated / total 
        elif self.device_type == DeviceType.MPS:
            return len(self.pressure_history) / self.max_history_size if self.pressure_history else 0
        else:
            return 0.5
