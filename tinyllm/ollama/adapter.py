from typing import Optional
import torch 
from ..models.transformer import TransformerConfig
from ..pipeline.generator import GenerationConfig
from .client import OllamaClient, OllamaConfig

class OllamaAdapter:
    def __init__(
        self,
        model_name: str,
        ollama_config: Optional[OllamaConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = ollama_config or OllamaConfig(model=model_name)
        self.client = OllamaClient(self.config)
        self.device = device or torch.device('cpu')

        model_info = self.client.get_model_info()
        self.model_config = self._map_model_config(model_info)
    
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        response = self.client.generate(
            prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens
        )
        return response["response"]
    
    def _map_model_config(self, model_info: dict) -> TransformerConfig:
        """Map Ollama model config to our engine config"""
        return TransformerConfig(
            num_layers=model_info.get("num_layers", 12),
            hidden_size=model_info.get("hidden_size", 2048),
            num_attention_heads=model_info.get("num_attention_heads", 32),
            max_sequence_length=self.config.context_size
        )