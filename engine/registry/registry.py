from pathlib import Path 
from typing import Dict, Optional, Union, List
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
    config: dict
    checkpoint_path: str
    model_type: str = "native"
    model_class: Optional[str] = None
    description: str = ""
    created_at: str = str(datetime.now())
    author: Optional[str] = None
    tags: List[str] = None

class ModelRegistry:
    def __init__(self, models_dir: Union[str, Path] = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models: Dict[str, ModelInfo] = {}
        self._load_models()

    def register_model(
        self,
        name: str,
        version: str,
        config_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        model_type: str = "native",
        model_class: Optional[str] = None,
        description: str = "",
        author: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        if model_type == "custom":
            if not model_class:
                raise ValueError("model_class required for custom models")
            try:
                module_path, class_name = model_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                getattr(module, class_name)
            except Exception as e:
                raise ValueError(f"Invalid model_class: {e}")

        model_dir = self.models_dir / f"{name}-{version}"
        model_dir.mkdir(exist_ok=True)

        with open(config_path) as f:
            config = json.load(f)

        new_checkpoint_path = model_dir / "model.pt"
        shutil.copy2(checkpoint_path, new_checkpoint_path)

        model_info = ModelInfo(
            name=name,
            version=version,
            config=config,
            checkpoint_path=str(new_checkpoint_path),
            model_type=model_type,
            model_class=model_class,
            description=description,
            author=author,
            tags=tags or []
        )

        model_id = f"{name}-{version}"
        self.models[model_id] = model_info

        info_path = model_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info.__dict__, f, indent=2)

        return model_info
    
    
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