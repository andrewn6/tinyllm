""" Test our sample models="""
import os
import sys
import torch
import uvicorn
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tinyllm.models.transformer import Transformer, TransformerConfig
from tinyllm.pipeline.tokenizer import Tokenizer, TokenizerConfig
from tinyllm.pipeline.generator import TextGenerator
from tinyllm.runtime.server import app

def initialize_app():
   device = torch.device("cuda" if torch.cuda.is_available() else 
                        "mps" if torch.backends.mps.is_available() else "cpu")
   print(f"Using device: {device}")
   
   model_config = TransformerConfig()
   tokenizer_config = TokenizerConfig()
   
   model = Transformer(model_config)
   tokenizer = Tokenizer(tokenizer_config)
   generator = TextGenerator(model, tokenizer, device=device)
   
   app.state.generator = generator
   app.state.device = device

if __name__ == "__main__":
   initialize_app()
   uvicorn.run(app, host="0.0.0.0", port=8000)
