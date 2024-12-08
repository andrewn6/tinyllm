import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any
from engine.models.base import BaseLLM, ModelConfig
from engine.pipeline.tokenizer import Tokenizer, TokenizerConfig

class TinyLLM(BaseLLM):
    """Tiny model for testing - 8M params"""
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
        
        # Initialize tokenizer explicitly
        tokenizer_config = TokenizerConfig(
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_seq_length
        )
        self.tokenizer = Tokenizer(tokenizer_config)
        
        # Model layers
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
    ) -> torch.Tensor:
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        # Get sequence length from reshaped input
        seq_length = input_ids.size(-1)
        
        # Embed tokens
        hidden_states = self.embedding(input_ids)
        
        # Add positional embeddings
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        
        # Process through transformer layers
        past_key_values = past_key_values or [None] * self.config.num_layers
        for i, layer in enumerate(self.layers):
            hidden_states, past_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i]
            )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits

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
        # Self attention
        attn_out, past_kv = self.attention(
            self.ln1(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        return x, past_kv

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_heads)
        self.qkv = nn.Linear(config.hidden_size, 3 * self.num_heads * self.head_dim)
        self.proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2),
            qkv
        )
        
        # Use past key values if provided
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        
        # Output
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