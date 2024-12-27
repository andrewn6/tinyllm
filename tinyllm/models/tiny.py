import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any
from tinyllm.models.base import BaseLLM, ModelConfig
from tinyllm.pipeline.tokenizer import Tokenizer, TokenizerConfig

class TinyLLM(BaseLLM):
    def __init__(self):
        config = ModelConfig(
            hidden_size=512,
            num_heads=8,
            num_layers=6,
            vocab_size=256,  
            max_seq_length=128,
            head_dim=64
        )
        super().__init__(config)
        
        tokenizer_config = TokenizerConfig(
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_seq_length
        )
        self.tokenizer = Tokenizer(tokenizer_config)
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        seq_length = input_ids.size(-1)
        
        hidden_states = self.embedding(input_ids)
        
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        hidden_states = hidden_states + self.position_embedding(position_ids)

        past_key_values = past_key_values or [None] * self.config.num_layers
        present_key_values = []
        
        for i, layer in enumerate(self.layers):
            hidden_states, past_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i]
            )
            present_key_values.append(past_kv)
        
        hidden_states = self.ln_f(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        return logits, present_key_values

class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = SelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, past_kv = self.attention(
            self.ln1(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        x = x + attn_out
        
        x = x + self.mlp(self.ln2(x))
        return x, past_kv

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_heads)
        self.hidden_size = config.hidden_size
        
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape
        
        q = self.q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = L
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            kv_seq_len = k.size(-2)
            
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(1).expand(B, 1, L, kv_seq_len)
            attn = attn.masked_fill(expanded_mask == 0, float('-inf'))
            
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        out = self.proj(out)
        
        return out, (k, v)

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))