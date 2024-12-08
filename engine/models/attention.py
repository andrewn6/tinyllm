import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass
from engine.compute.kernels import KernelManager, KernelConfig
from engine.memory.cache import PagedKVCache

@dataclass
class AttentionConfig:
    num_heads: int = 32 
    head_dim: int = 64
    dropout: float = 0.1
    max_seq_length: int = 2048
    use_kv_cache: bool = True

class Attention(nn.Module):
    def __init__(
            self,
            config: AttentionConfig,
            kernel_manager: Optional[KernelManager] = None
    ):
        super().__init__()
        self.config = config

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.num_heads * config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        if kernel_manager is None:
            kernel_config = KernelConfig(
                num_heads=config.num_heads,
                head_size=config.head_dim
            )
            kernel_manager = KernelManager(kernel_config)
        self.kernel_manager = kernel_manager

        self.kv_cache = None
        if config.use_kv_cache:
            self.kv_cache = PagedKVCache(
                    num_layers=1,
                    num_heads=config.num_heads,
                    head_size=config.head_dim,
                    device=self.kernel_manager.device
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[int] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        if batch_size == 0:
            hidden_states = hidden_states.unsqueeze(0)
            batch_size = 1
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
 
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]

        if self.kv_cache is not None and use_cache:
            if past_key_values is None and sequence_id is not None:
                self.kv_cache.allocate_blocks(
                        sequence_id,
                        num_tokens=self.config.max_seq_length
                )
                past_key_values = None
            elif past_key_values is not None:
                key_states = torch.cat([past_key_values[0], key_states], dim=2)
                value_states = torch.cat([past_key_values[1], value_states], dim=2)
                kv_seq_len = key_states.shape[-2]

            if sequence_id is not None:
                self.kv_cache.update_kv_cache(
                        sequence_id,
                        layer_id=0,
                        position=kv_seq_len - 1,
                        key=key_states[:, :, -1:],
                        value=value_states[:, :, -1:]
                )

        if attention_mask is not None:
            attention_mask = attention_mask.view(
                batch_size, 1, seq_length, kv_seq_len
            ).expand(-1, self.num_heads, -1, -1)

        attn_output = self.kernel_manager.get_attention_kernel().attention(
                query_states,
                key_states,
                value_states,
                mask=attention_mask,
                dropout_p=self.config.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.hidden_size
        )

        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        if use_cache:
            past_key_value = (key_states, value_states)
            return attn_output, past_key_value
        
        return attn_output, None
    
    def clear_cache(self, sequence_id: Optional[int] = None):
        if self.kv_cache is not None:
            if sequence_id is not None:
                self.kv_cache.free_sequence(sequence_id)
            else:
                self.kv_cache.clear()
