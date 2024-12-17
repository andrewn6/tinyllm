import click 
import uvicorn
import torch
from ..runtime.server import app 
from ..registry.registry import ModelRegistry
from ..models.transformer import Transformer, TransformerConfig
from ..pipeline.tokenizer import Tokenizer, TokenizerConfig
from ..pipeline.generator import TextGenerator, GenerationConfig
import os
from ..metrics.prometheus import PrometheusConfig
import time

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

@cli.group()
def metrics():
    """Metrics management commands"""
    pass

@metrics.command()
@click.option('--port', default=8001, help='Port for Prometheus metrics server')
def start(port):
    """Start Prometheus metrics server standalone"""
    from prometheus_client import start_http_server
    click.echo(f"Starting Prometheus metrics server on port {port}")
    start_http_server(port)
    click.echo("Metrics server running. Press Ctrl+C to stop.")
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

# Model commands
@model.command()
@click.argument('name')
@click.argument('version')
@click.option('--checkpoint', required=True, type=click.Path(exists=True))
@click.option('--config', required=True, type=click.Path(exists=True))
@click.option('--description', default='')
@click.option('--enable-metrics/--no-metrics', default=False,
              help='Enable metrics collection for this model')
@click.option('--metrics-port', default=8001,
              help='Default metrics port for this model')
def register(name, version, checkpoint, config, description, 
            enable_metrics, metrics_port):
    """Register a model in the registry"""
    registry = ModelRegistry()
    
    # Add metrics configuration
    metrics_config = {
        'enabled': enable_metrics,
        'port': metrics_port
    }
    
    model_info = registry.register_model(
        name=name,
        version=version,
        checkpoint_path=checkpoint,
        config_path=config,
        description=description,
        metrics_config=metrics_config
    )
    
    click.echo(f"Registered model {name} v{version}")
    if enable_metrics:
        click.echo(f"Metrics enabled on port {metrics_port}")

@model.command()
@click.option('--show-metrics/--no-metrics', default=False,
              help='Show metrics configuration')
def list(show_metrics):
    """List all registered models"""
    registry = ModelRegistry()
    for model_id, info in registry.list_models().items():
        click.echo(f"\n{model_id}:")
        click.echo(f"  Checkpoint: {info.checkpoint_path}")
        click.echo(f"  Description: {info.description}")
        
        if show_metrics:
            # Show metrics configuration if available
            metrics_config = getattr(info, 'metrics_config', {})
            if metrics_config:
                click.echo("  Metrics:")
                click.echo(f"    Enabled: {metrics_config.get('enabled', False)}")
                click.echo(f"    Port: {metrics_config.get('port', 8001)}")

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
@click.option('--enable-metrics/--no-metrics', default=False, 
              help='Enable Prometheus metrics collection')
@click.option('--metrics-port', default=8001, 
              help='Port for Prometheus metrics endpoint')
@click.option('--metrics-path', default='/metrics',
              help='Path for Prometheus metrics endpoint')
def serve(host, port, model_name, model_version, enable_metrics, 
          metrics_port, metrics_path):
    """Start the TinyLLM server"""
    os.environ['MODEL_NAME'] = model_name
    if model_version:
        os.environ['MODEL_VERSION'] = model_version
    
    # Configure metrics
    os.environ['ENABLE_METRICS'] = str(enable_metrics).lower()
    os.environ['METRICS_PORT'] = str(metrics_port)
    os.environ['METRICS_PATH'] = metrics_path
    
    click.echo(f"Starting server with model: {model_name}")
    if enable_metrics:
        click.echo(f"Prometheus metrics enabled:")
        click.echo(f"  - Endpoint: http://{host}:{metrics_port}{metrics_path}")
        click.echo(f"  - Scrape interval: 15s")
    
    uvicorn.run(app, host=host, port=port)