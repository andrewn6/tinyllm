from pathlib import Path 
from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass
from datetime import datetime
import json 
import os
import shutil
import importlib

@dataclass 
class ModelInfo:
    name: str
    version: str
    checkpoint_path: str
    config_path: str
    description: str = ""
    metrics_config: Optional[Dict[str, Any]] = None

class ModelRegistry:
    def __init__(self):
        self.models_dir = Path.home() / ".tinyllm" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelInfo] = {}
        self._load_models()
    
    def register_model(self,
                      name: str,
                      version: str,
                      checkpoint_path: str,
                      config_path: str,
                      description: str = "",
                      metrics_config: Optional[Dict[str, Any]] = None) -> ModelInfo:
        """Register a model in the registry"""
        model_id = f"{name}-{version}"
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Copy files to registry
        new_checkpoint = model_dir / "model.pt"
        new_config = model_dir / "config.json"
        
        shutil.copy2(checkpoint_path, new_checkpoint)
        shutil.copy2(config_path, new_config)
        
        # Create model info
        info = ModelInfo(
            name=name,
            version=version,
            checkpoint_path=str(new_checkpoint),
            config_path=str(new_config),
            description=description,
            metrics_config=metrics_config
        )
        
        # Save info
        with open(model_dir / "info.json", "w") as f:
            json.dump(info.__dict__, f, indent=2)
        
        self.models[model_id] = info
        return info
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelInfo]:
        if version:
            return self.models.get(f"{name}-{version}")
        versions = [k for k in self.models.keys() if k.startswith(f"{name}-")]
        if not versions:
            return None
        return self.models[sorted(versions)[-1]]

    def _load_models(self):
        for model_dir in self.models_dir.glob("*-*"):
            if not model_dir.is_dir():
                continue 
            info_path = model_dir / "info.json"
            if info_path.exists():
                with open(info_path) as f:
                    data = json.load(f)
                    self.models[model_dir.name] = ModelInfo(**data)