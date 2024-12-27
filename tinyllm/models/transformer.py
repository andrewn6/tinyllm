import torch
import torch.nn as nn
from .attention import Attention, AttentionConfig
from .ffn import FFN, FFNConfig
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass
from tinyllm.compute.kernels import KernelManager

@dataclass
class TransformerConfig:
    n_head: int
    hidden_size: int = 2048
    num_layers: int = 32 
    max_sequence_length: int = 2048
    vocab_size: int = 32000
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    architecture: str = "default"  # Add architecture type to support variants

class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        kernel_manager: Optional[KernelManager] = None
    ):
        super().__init__()
        self.config = config

        if config.architecture == "default":
            head_dim = config.hidden_size // config.n_head

            self.input_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)

            self.input_layernorm = nn.LayerNorm(
                    config.hidden_size,
                    eps=config.layer_norm_epsilon
            )

            self.post_attention_layernorm = nn.LayerNorm(
                    config.hidden_size,
                    eps=config.layer_norm_epsilon
            )

            attention_config = AttentionConfig(
                    num_heads=config.n_head,
                    head_dim=head_dim,
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

        elif config.architecture == "variant1":
            self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.attention = nn.ModuleDict({
                'qkv': nn.Linear(config.hidden_size, 3 * config.hidden_size),
                'proj': nn.Linear(config.hidden_size, config.hidden_size)
            })
            self.mlp = nn.ModuleDict({
                'fc1': nn.Linear(config.hidden_size, 4 * config.hidden_size),
                'fc2': nn.Linear(4 * config.hidden_size, config.hidden_size)
            })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[int] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        if self.config.architecture == "default":
            return self._forward_default(input_ids, attention_mask, sequence_id, past_key_values, use_cache)
        elif self.config.architecture == "variant1":
            return self._forward_variant1(input_ids, attention_mask, sequence_id, past_key_values, use_cache)

    def _forward_default(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[int] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states = input_ids 

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output, present_key_value = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                sequence_id=sequence_id,
                past_key_values=past_key_values,
                use_cache=use_cache
        )

        hidden_states = residual + attention_output
        
        # feed forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states, None

    def _forward_variant1(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[int] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        qkv = self.attention['qkv'](hidden_states)
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, self.config.n_head, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=q.dtype)
            if sequence_id is not None:
                seq_mask = attention_mask.new_ones(attention_mask.size()) * float('-inf')
                seq_mask[:, :, :, :seq_len] = attention_mask
                attention_mask = seq_mask
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        hidden_states = self.attention['proj'](attn_output)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp['fc1'](hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.mlp['fc2'](hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, (k, v)
        return hidden_states, None

class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        kernel_manager: Optional[KernelManager] = None
    ):
        super().__init__()
        self.config = config

        if config.architecture == "default":
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList([
                TransformerLayer(config, kernel_manager) for _ in range(config.num_layers)
            ])
            self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        elif config.architecture == "variant1":
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
            self.layers = nn.ModuleList([
                TransformerLayer(config, kernel_manager) for _ in range(config.num_layers)
            ])
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if self.config.architecture == "default":
            return self._forward_default(input_ids, attention_mask, **kwargs)
        elif self.config.architecture == "variant1":
            return self._forward_variant1(input_ids, attention_mask, **kwargs)

    def _forward_default(
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
            layer_past = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_key_value = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sequence_id=sequence_id,
                    past_key_values=layer_past,
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

    def _forward_variant1(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[int] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states = self.embedding(input_ids)
        
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
        hidden_states = hidden_states + self.position_embedding(position_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                sequence_id=sequence_id,
                past_key_values=layer_past,
                use_cache=use_cache
            )

            if use_cache:
                present_key_values.append(present_key_value)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, present_key_values
