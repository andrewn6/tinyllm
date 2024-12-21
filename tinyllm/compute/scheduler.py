import threading
import time
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from enum import Enum
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor
from tinyllm.memory.cache import get_best_device, DeviceType, get_device_type

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
        max_batch_size: int = 32,
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

    def update_pressure_history(self, pressure: float):
        self.pressure_history.append(pressure)
        if len(self.pressure_history) > self.max_history_size:
            self.pressure_history.pop(0)

    def should_reduce_batch(self) -> bool:
        pressure = self.get_memory_pressure()
        self.update_pressure_history(pressure)

        if pressure > self.memory_threshold:
            return True

        if len(self.pressure_history) > 10:
            recent_trend = sum(self.pressure_history[-10:])
            if recent_trend > self.warning_threshold:
                return True

        return False
    
    def can_increase_batch(self) -> bool:
        pressure = self.get_memory_pressure()
        return pressure < self.warning_threshold

"""
Operation scheduler with dynamic batching & memory-aware execution
"""
class Scheduler:
    def __init__(
            self,
            batch_config: Optional[BatchConfig] = None,
            device: Optional[torch.device] = None,
            num_workers: int = 4
    ):
        self.device = device if device is not None else get_best_device()
        self.batch_config = batch_config or BatchConfig()
        self.memory_tracker = MemoryTracker(self.device)

        self.high_priority_queue = PriorityQueue()
        self.normal_queue = PriorityQueue()

        self.current_batch: Dict[OperationType, List[ScheduledOperation]] = {
            op_type: [] for op_type in OperationType
        }
        self.batch_lock = threading.Lock()

        self.workers = ThreadPoolExecutor(max_workers=num_workers)

        self.active_operations: Dict[str, ScheduledOperation] = {}
        self.operation_lock = threading.Lock()
        
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

    def schedule_operation(
        self,
        op_type: OperationType,
        callback: Callable,
        data: Any,
        batch_size: int = 1,
        seq_length: int = 0,
        priority: int = 1
    ) -> str:
        operation = ScheduledOperation(
                priority=priority,
                timestamp=time.time(),
                op_type=op_type,
                batch_size=batch_size,
                seq_length=seq_length,
                callback=callback,
                data=data
        )

        with self.operation_lock:
            self.active_operations[operation.id] = operation
        
        if priority > 1:
            self.active_operations[operation.id] = operation

        else:
            self.normal_queue.put(operation)

        return operation.id
    
    def _scheduler_loop(self):
        while self.running:
            self._process_queues()
            self._check_memory_pressure()
            time.sleep(0.001)

    def _process_queues(self):
        while not self.high_priority_queue.empty():
            operation = self.high_priority_queue.get()
            self._try_batch_operation(operation)

        while not self.normal_queue.empty():
            operation = self.normal_queue.get()
            self._try_batch_operation(operation)

        self._execute_ready_batches()

    def _try_batch_operation(self, operation: ScheduledOperation):
        with self.batch_lock:
            current_batch = self.current_batch[operation.op_type]

            if not current_batch:
                current_batch.append(operation)
                return
            
            total_batch_size = sum(op.batch_size for op in current_batch)
            max_seq_len = max(op.seq_length for op in current_batch)

            can_batch = (
                total_batch_size + operation.batch_size <= self.batch_config.current_max_batch
                and max(max_seq_len, operation.seq_length) <= self.batch_config.current_max_seq_len
            
            )

            if can_batch:
                current_batch.append(operation)
            else:
                self._execute_batch(operation.op_type)
                self.current_batch[operation.op_type] = [operation]

    def _execute_ready_batches(self):
        with self.batch_lock:
            for op_type in OperationType:
                batch = self.current_batch[op_type]
                if not batch:
                    continue
                
                oldest_op = min(batch, key=lambda x: x.timestamp)
                if time.time() - oldest_op.timestamp > self.batch_config.batch_timeout_ms / 1000:
                    self._execute_batch(op_type)

    def _execute_batch(self, op_type: OperationType):
        with self.batch_lock:
           batch = self.current_batch[op_type]
           if not batch: 
               return

           self.current_batch[op_type] = []

        try:
            if self.device.type == "cuda":
                with torch.cuda.stream(self.stream):
                    self._execute_operations(batch)
                self.stream.synchronize()
            else:
                self._execute_operations(batch)

        except Exception as e:
            print(f"Error executing batch: {e}")
            for op in batch:
                self._handle_failed_operation(op)

    def _execute_operations(self, operations: List[ScheduledOperation]):
        for op in operations:
            try:
                result = op.callback(op.data)
                self._handle_completed_operation(op, result)
            except Exception as e:
                self._handle_failed_operation(op, err=str(e))

    def _handle_completed_operation(self, operation: ScheduledOperation, result: any):
        with self.operation_lock:
            if operation.id in self.active_operations:
                del self.active_operations[operation.id]

    def _handle_failed_operation(self, operation: ScheduledOperation, error: str = ""):
        with self.operation_lock:
            if operation.id in self.active_operations:
                del self.active_operations[operation.id]
        print(f"Operation {operation.id} failed: {error}")

    def _check_memory_pressure(self):
        if self.memory_tracker.should_reduce_batch():
            self.batch_config.current_max_batch = max(\
                    1,
                    self.batch_config.current_max_batch // 2
            )
        elif self.memory_tracker.can_increase_batch():
            self.batch_config.current_max_batch = min(
                self.batch_config.max_batch_size,
                self.batch_config.current_max_batch * 2
            )

    def get_stats(self) -> Dict:
        return {
                "active_operations": len(self.active_operations),
                "high_priority_queue_size": self.high_priority_queue.qsize(),
                "normal_queue_size": self.normal_queue.qsize(),
                "current_batch_size": self.batch_config.current_max_batch,
                "memory_pressure": self.memory_tracker.get_memory_pressure()
        }
    
    def shutdown(self):
        self.running = False
        self.scheduler_thread.join()
        self.workers.shutdown()
