from .base import (
    BaseLLM,
    ModelConfig
)

from .tiny import TinyLLM

from .transformer import (
    TransformerLayer,
    TransformerConfig
)

# Default configurations
DEFAULT_CONFIG = ModelConfig(
    hidden_size=512,
    num_heads=8,
    num_layers=6,
    vocab_size=256,
    max_seq_length=128,
    head_dim=64
)

__all__ = [
    # Base classes
    'BaseLLM',
    'ModelConfig',
    
    # Model implementations
    'TinyLLM',
    
    # Components
    'TransformerLayer',
    'TransformerConfig',
    
    # Defaults
    'DEFAULT_CONFIG'
]