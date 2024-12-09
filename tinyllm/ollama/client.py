import requests
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    context_size: int = 2048
    num_gpu: int = 1
    num_thread: int = 4


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.base_url = config.base_url
        self.model = config.model
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "context": kwargs.get("context", []),
                "options": {
                    "num_gpu": self.config.num_gpu,
                    "num_thread": self.config.num_thread
                }
            }
        )
        return response.json()

    def get_model_info(self) -> Dict:
        response = requests.get(
            f"{self.base_url}/api/show",
            params={"name": self.model}
        )
        return response.json()

    def load_model(self) -> Dict:
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": self.model}
        )
        return response.json()