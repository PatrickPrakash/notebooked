"""Main CLI entry point for notebooked"""

import click
import sys
import yaml
from pathlib import Path
from .core.models import ProjectConfig
from .core.parser import NotebookParser
from .core.generator import CodeGenerator
from .providers.sagemaker import SageMakerProvider
from .providers.local import LocalProvider


def load_config(config_path: str) -> ProjectConfig:
    """Load and validate configuration"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        
    return ProjectConfig(**data)


@click.group()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.pass_context
def main(ctx, config):
    """Notebooked: Convert and deploy Jupyter notebooks"""
    ctx.ensure_object(dict)
    ctx.obj['CONFIG_PATH'] = config


@main.command()
def init():
    """Initialize a new notebooked project"""
    config_path = Path("config.yaml")
    if config_path.exists():
        click.echo("config.yaml already exists.")
        return
        
    sample_config = {
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "default"
        },
        "aws": {
            "region": "us-east-1"
        },
        "experiments": [
            {
                "name": "example-experiment",
                "notebook": "notebooks/example.ipynb",
                "data_path": "data/",
                "hyperparameters": {
                    "epochs": 10,
                    "batch_size": 32
                }
            }
        ]
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f, sort_keys=False)
        
    click.echo("Initialized config.yaml")


@main.command()
@click.argument('experiment_name')
@click.pass_context
def convert(ctx, experiment_name):
    """Convert a notebook to Python scripts"""
    config_path = ctx.obj['CONFIG_PATH']
    try:
        config = load_config(config_path)
        exp_config = config.get_experiment(experiment_name)
        
        click.echo(f"Converting notebook for experiment: {experiment_name}")
        
        # Parse notebook
        parser = NotebookParser(str(exp_config.notebook))
        parser.parse()
        
        # Validate tags
        issues = parser.validate_tags()
        if issues['invalid_tags']:
            click.echo(f"Warning: Invalid tags found: {issues['invalid_tags']}")
        if issues['missing_sections']:
            click.echo(f"Error: Missing required sections: {issues['missing_sections']}")
            sys.exit(1)
            
        # Extract code
        extracted = parser.extract_tagged_code()
        
        # Generate scripts
        generator = CodeGenerator(output_dir="generated")
        generated_files = generator.generate_all(
            extracted=extracted,
            experiment_name=experiment_name,
            hyperparameters=exp_config.hyperparameters,
            mlflow_config={'tracking_uri': config.mlflow.tracking_uri, 'experiment_name': config.mlflow.experiment_name}
        )
        
        click.echo("Generated files:")
        for name, path in generated_files.items():
            click.echo(f"  {name}: {path}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('experiment_name')
@click.option('--provider', type=click.Choice(['sagemaker', 'local']), default='sagemaker', help='Compute provider')
@click.option('--wait/--no-wait', default=True, help='Wait for training to complete')
@click.pass_context
def train(ctx, experiment_name, provider, wait):
    """Run training"""
    config_path = ctx.obj['CONFIG_PATH']
    try:
        config = load_config(config_path)
        exp_config = config.get_experiment(experiment_name)
        
        # Ensure scripts are generated
        ctx.invoke(convert, experiment_name=experiment_name)
        
        # Initialize provider
        if provider == 'sagemaker':
            impl = SageMakerProvider(
                region=config.aws.region,
                role=config.aws.role,
                access_key=config.aws.access_key,
                secret_key=config.aws.secret_key
            )
        elif provider == 'local':
            impl = LocalProvider()
        
        source_dir = Path("generated") / experiment_name
        
        click.echo(f"Starting training for {experiment_name} on {provider}...")
        result = impl.train(
            experiment_name=experiment_name,
            source_dir=source_dir,
            data_path=exp_config.data_path,
            hyperparameters=exp_config.hyperparameters,
            instance_type=exp_config.instance_type,
            instance_count=exp_config.instance_count,
            wait=wait
        )
        
        click.echo(f"Training job submitted: {result['job_name']}")
        if wait:
            click.echo(f"Model URI: {result.get('model_uri')}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('experiment_name')
@click.option('--model-uri', help='S3 URI of the model artifact')
@click.option('--endpoint-name', help='Name for the endpoint')
@click.option('--serverless', is_flag=True, help='Deploy to serverless endpoint')
@click.option('--provider', type=click.Choice(['sagemaker', 'local']), default='sagemaker', help='Compute provider')
@click.pass_context
def deploy(ctx, experiment_name, model_uri, endpoint_name, serverless, provider):
    """Deploy model to endpoint"""
    config_path = ctx.obj['CONFIG_PATH']
    try:
        config = load_config(config_path)
        exp_config = config.get_experiment(experiment_name)
        
        if not endpoint_name:
            endpoint_name = f"{experiment_name}-endpoint"
            
        # Initialize provider
        if provider == 'sagemaker':
            impl = SageMakerProvider(
                region=config.aws.region,
                role=config.aws.role,
                access_key=config.aws.access_key,
                secret_key=config.aws.secret_key
            )
        elif provider == 'local':
            impl = LocalProvider()
        
        if not model_uri:
            click.echo("Error: --model-uri is required")
            sys.exit(1)
            
        click.echo(f"Deploying {endpoint_name} to {provider}...")
        result = impl.deploy(
            model_uri=model_uri,
            endpoint_name=endpoint_name,
            instance_type=exp_config.instance_type,
            instance_count=1,
            serverless=serverless
        )
        
        click.echo(f"Endpoint deployed: {result['endpoint_name']}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--provider', type=click.Choice(['local', 'sagemaker']), default='local', help='Target provider for the CI pipeline')
@click.option('--branch', default='main', help='Branch to trigger the workflow')
def generate_workflow(provider, branch):
    """Generate GitHub Actions workflow"""
    from .core.workflow import WorkflowGenerator
    
    try:
        generator = WorkflowGenerator()
        path = generator.generate(provider=provider, branch=branch)
        click.echo(f"Generated GitHub Actions workflow at: {path}")
    except Exception as e:
        click.echo(f"Error generating workflow: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
