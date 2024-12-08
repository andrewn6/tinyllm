import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from ..memory.cache import get_device_type, DeviceType

@dataclass
class KernelConfig:
    dtype: torch.dtype = torch.float16
    max_context_length: int = 2048
    head_size: int = 64
    num_heads: int = 16

def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class AttentionKernels:
    def __init__(
        self, 
        config: KernelConfig,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device or get_optimal_device()
        self.device_type = get_device_type(self.device)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0
    ) -> torch.Tensor:
        # Scale query
        scaling = float(q.size(-1)) ** -0.5
        q = q * scaling

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1))

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn = F.softmax(attn, dim=-1)

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        # Compute output
        output = torch.matmul(attn, v)
        return output

class KernelManager:
    def __init__(
        self,
        config: Optional[KernelConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or KernelConfig()
        self.device = device or get_optimal_device()
        self.attention = AttentionKernels(self.config, self.device)

    def get_attention_kernel(self) -> AttentionKernels:
        return self.attention

    def get_device_capabilities(self) -> dict:
        capabilities = {
            "device": str(self.device),
            "device_type": self.device_type.value,
            "dtype": str(self.config.dtype)
        }
        
        if self.device_type == DeviceType.CUDA:
            capabilities.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(self.device)
            })
        elif self.device_type == DeviceType.MPS:
            capabilities.update({
                "mps_version": torch.__version__,
                "platform": "Apple Silicon"
            })
            
        return capabilities
