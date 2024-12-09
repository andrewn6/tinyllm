import click 
import uvicorn
import torch
from ..runtime.server import app 
from ..registry.registry import ModelRegistry
from ..models.transformer import Transformer, TransformerConfig
from ..pipeline.tokenizer import Tokenizer, TokenizerConfig
from ..pipeline.generator import TextGenerator, GenerationConfig
import os

@click.group()
def cli():
    """TinyLLM CLI - Development-focused inference engine"""
    pass

@cli.group()
def model():
    """Model management commands"""
    pass

@cli.group()
def generate():
    """Text generation commands"""
    pass

# Model commands
@model.command()
@click.argument('name')
@click.argument('version')
@click.option('--checkpoint', required=True, type=click.Path(exists=True))
@click.option('--config', required=True, type=click.Path(exists=True))
@click.option('--description', default='')
def register(name, version, checkpoint, config, description):
    """Register a model in the registry"""
    registry = ModelRegistry()
    model_info = registry.register_model(
        name=name,
        version=version,
        checkpoint_path=checkpoint,
        config_path=config,
        description=description
    )
    click.echo(f"Registered model {name} v{version}")

@model.command()
def list():
    """List all registered models"""
    registry = ModelRegistry()
    for model_id, info in registry.list_models().items():
        click.echo(f"\n{model_id}:")
        click.echo(f"  Checkpoint: {info.checkpoint_path}")
        click.echo(f"  Description: {info.description}")

# Generate commands
@generate.command()
@click.argument('model_name')
@click.option('--prompt', required=True, help='Text prompt for generation')
@click.option('--max-tokens', default=100, help='Maximum tokens to generate')
@click.option('--temperature', default=0.7, help='Sampling temperature')
@click.option('--top-k', default=50, help='Top-k sampling parameter')
@click.option('--top-p', default=0.9, help='Top-p sampling parameter')
def text(model_name, prompt, max_tokens, temperature, top_k, top_p):
    """Generate text with a registered model"""
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    
    registry = ModelRegistry()
    model_info = registry.get_model(model_name)
    if not model_info:
        raise click.ClickException(f"Model {model_name} not found in registry")
    
    click.echo(f"Loading model from: {model_info.checkpoint_path}")
    
    # Initialize model
    model_config = TransformerConfig(**model_info.config)
    model = Transformer(model_config)
    model.load_state_dict(torch.load(model_info.checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer_config = TokenizerConfig(
        vocab_size=model_config.vocab_size,
        max_sequence_length=model_config.max_sequence_length
    )
    tokenizer = Tokenizer(tokenizer_config)
    
    # Initialize generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Generate text
    config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    click.echo(f"Generating with model {model_name}...")
    with torch.no_grad():
        output = generator.generate(prompt, config)
    click.echo("\nGenerated text:")
    click.echo(output)

# Server command
@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
@click.option('--model-name', required=True)
@click.option('--model-version', default=None)
def serve(host, port, model_name, model_version):
    """Start the TinyLLM server"""
    os.environ['MODEL_NAME'] = model_name
    if model_version:
        os.environ['MODEL_VERSION'] = model_version
    
    click.echo(f"Starting server with model: {model_name}")
    uvicorn.run(app, host=host, port=port)