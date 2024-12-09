import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class FFNConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    dropout: float = 0.1
    activation: str = "gelu"

class FFN(nn.Module):
    def __init__(self, config: FFNConfig):
        super().__init__()
        self.config = config

        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

        # activate
        self.act = nn.GELU() if config.activation == "gelu" else nn.ReLU()

        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
         x: Input tensor

        Output:
            Output tensor of shape
        """

        h = self.act(self.up_proj(x))

        out = self.dropout(self.down_proj(h))
        
        return out
