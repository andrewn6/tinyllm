import torch
import torch.nn as nn
from .attention import Attention, AttentionConfig
from .ffn import FFN, FFNConfig
from typing import Tuple, Optional, List
from dataclasses import dataclass
from engine.compute.kernels import KernelManager

@dataclass
class TransformerConfig:
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_layers: int = 32 
    max_sequence_length: int = 2048
    vocab_size: int = 32000
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True

class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        kernel_manager: Optional[KernelManager] = None
    ):
        super().__init__()
        self.config = config

        self.input_layernorm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon
        )

        self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon
        )

        attention_config = AttentionConfig(
                num_heads=config.num_attention_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                dropout=config.dropout,
                max_seq_length=config.max_sequence_length,
                use_kv_cache=config.use_cache
        )
        self.attention = Attention(attention_config, kernel_manager)

        ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size * 4,
                dropout=config.dropout
        )
        self.ffn = FFN(ffn_config)


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[int] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output, present_key_value = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                sequence_id=sequence_id,
                past_key_value=past_key_value,
                use_cache=use_cache
        )

        hidden_states = residual + attention_output
        
        # feed forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden-states, present_key_value
        return hidden_states, None

class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        kernel_manager: Optional[KernelManager] = None
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            TransformerLayer(config, kernel_manager)
            for _ in range(config.num_layers)
        ])

        self.final_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon
        )

        self.output_proj = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[int] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_key_value = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sequence_id=sequence_id,
                    past_key_value=past_key_value,
                    use_cache=use_cache
            )

            if use_cache:
                present_key_values.append(present_key_value)


        hidden_states = self.final_layernorm(hidden_states)
        logits = self.output_proj(hidden_states)

        return logits, present_key_values
    
    def clear_cache(self, sequence_id: Optional[int] = None):
        for layer in self.layers:
            layer.attention.clear_cache(sequence_id)
