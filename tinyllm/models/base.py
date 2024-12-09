from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass

from tinyllm.memory.cache import PagedKVCache
from tinyllm.compute.kernels import KernelManager
from tinyllm.pipeline.generator import GenerationConfig
from tinyllm.pipeline.tokenizer import Tokenizer, TokenizerConfig

@dataclass
class ModelConfig:
    hidden_size: int
    num_heads: int
    num_layers: int
    vocab_size: int
    max_seq_length: int
    head_dim: Optional[int] = None
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

class BaseLLM(nn.Module, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize tokenizer
        tokenizer_config = TokenizerConfig(
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_seq_length
        )
        self.tokenizer = Tokenizer(tokenizer_config)
        
        self.pad_token_id = getattr(config, 'pad_token_id', 0)
        self.eos_token_id = getattr(config, 'eos_token_id', 2)
        self.bos_token_id = getattr(config, 'bos_token_id', 1)
        
     
        self.kernel_manager = KernelManager()
        self.kv_cache = PagedKVCache(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_size=config.head_dim,
            max_seq_len=config.max_seq_length,
            dtype=torch.float16,
            device=None
        )
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        pass

    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs: Any
    ) -> Union[str, torch.Tensor]:
        if isinstance(prompt, str):
            input_ids = self.tokenize(prompt)
        else:
            input_ids = prompt
            
        config = generation_config or GenerationConfig()
        
        with torch.no_grad():
            # Initialize KV cache for generation
            self.kv_cache.reset()
            
            output_ids = self._generate_tokens(
                input_ids,
                max_length=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                **kwargs
            )
            
        if isinstance(prompt, str):
            return self.detokenize(output_ids)
        return output_ids

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        **kwargs: Any
    ) -> torch.Tensor:
        """Token generation with engine's KV cache"""
        batch_size = input_ids.shape[0]
        current_ids = input_ids
        
        # Pre-allocate KV cache
        self.kv_cache.allocate(batch_size)
        
        for i in range(max_length):
            past_kv = self.kv_cache.get_cache(i)
            
            with self.kernel_manager.optimize():
                outputs = self.forward(
                    current_ids[:, -1:] if i > 0 else current_ids,
                    past_key_values=past_kv,
                    **kwargs
                )
            
            logits = outputs[:, -1, :]
            
            if hasattr(outputs, 'past_key_values'):
                self.kv_cache.update(i, outputs.past_key_values)
            
            next_token = self._sample_next_token(
                logits,
                temperature=temperature,
                top_p=top_p
            )
            
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            if (next_token == self.eos_token_id).all():
                break
            if current_ids.shape[1] >= self.config.max_seq_length:
                break
                
        self.kv_cache.clear()
        return current_ids

    def prepare_for_inference(self) -> None:
        self.eval() 
        
        if hasattr(self, 'kv_cache'):
            self.kv_cache.preallocate()
        
        if torch.cuda.is_available():
            self.cuda()
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        if hasattr(self, 'kv_cache'):
            self.kv_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()

