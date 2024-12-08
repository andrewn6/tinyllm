from .generator import (
    TextGenerator,
    GenerationConfig
)

from .tokenizer import (
    Tokenizer,
    TokenizerConfig
)

# Default configurations
DEFAULT_GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    use_cache=True
)

DEFAULT_TOKENIZER_CONFIG = TokenizerConfig(
    vocab_size=256,  # Byte-level tokenization
    max_sequence_length=2048,
    pad_token_id=0,
    eos_token_id=2,
    bos_token_id=1,
    unk_token_id=3
)

__all__ = [
    # Generation components
    'TextGenerator',
    'GenerationConfig',
    
    # Tokenization components
    'Tokenizer',
    'TokenizerConfig',
    
    # Default configurations
    'DEFAULT_GENERATION_CONFIG',
    'DEFAULT_TOKENIZER_CONFIG'
]