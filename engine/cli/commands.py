import click 
from ..runtime.server import serve 
from ..registry.registry import ModelRegistry
import os

@click.group()
def cli():
    pass

@cli.group()
def model():
    pass

@model.command()
@click.argument('name')
@click.argument('version')
@click.option('--config', required=True, type=click.Path(exists=True))
@click.option('--checkpoint', required=True, type=click.Path(exists=True))
@click.option('--model-type', default='native')
@click.option('--description', default='')
def register(name, version, config, checkpoint, model_type, description):
    registry = ModelRegistry()
    model_info = registry.register_model(
        name=name,
        version=version,
        config_path=config,
        checkpoint_path=checkpoint,
        model_type=model_type,
        description=description
    )
    click.echo(f"Registered model {name} v{version}")

@model.command()
def list():
    registry = ModelRegistry()
    for model_id, info in registry.list_models().items():
        click.echo(f"\n{model_id}:")
        click.echo(f"  Type: {info.model_type}")
        click.echo(f"  Checkpoint: {info.checkpoint_path}")

@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
@click.option('--model-name', required=True)
@click.option('--model-type', default='native')
@click.option('--model-version', default=None)
def serve(host, port, model_name, model_type, model_version):
    os.environ['MODEL_TYPE'] = model_type
    os.environ['MODEL_NAME'] = model_name
    if model_version:
        os.environ['MODEL_VERSION'] = model_version
    
    click.echo(f"Starting server with {model_type} model: {model_name}")
    serve(host=host, port=port)