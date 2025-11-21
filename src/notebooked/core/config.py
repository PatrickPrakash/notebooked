"""Configuration loader for notebooked"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MLFlowConfig:
    """MLFlow configuration"""
    tracking_uri: str
    experiment_name: str = "default"


@dataclass
class AWSConfig:
    """AWS configuration"""
    region: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    role: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Single experiment configuration"""
    name: str
    notebook: str
    data_path: str
    description: str = ""
    mlflow_experiment: Optional[str] = None
    hyperparameters: Dict[str, Any] = None
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}


class ConfigLoader:
    """Load and validate pipeline configuration"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config: Dict = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration structure"""
        # Minimal validation
        if 'experiments' not in self._config:
            raise ValueError("No experiments defined in configuration")
    
    @property
    def mlflow(self) -> MLFlowConfig:
        """Get MLFlow configuration"""
        mlflow_data = self._config.get('mlflow', {})
        return MLFlowConfig(
            tracking_uri=mlflow_data.get('tracking_uri', 'http://localhost:5000'),
            experiment_name=mlflow_data.get('experiment_name', 'default')
        )
    
    @property
    def aws(self) -> AWSConfig:
        """Get AWS configuration"""
        aws_data = self._config.get('aws', {})
        return AWSConfig(
            region=os.environ.get('AWS_REGION', aws_data.get('region', 'us-east-1')),
            access_key=os.environ.get('AWS_ACCESS_KEY_ID', aws_data.get('access_key')),
            secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY', aws_data.get('secret_key')),
            role=os.environ.get('SAGEMAKER_ROLE', aws_data.get('role'))
        )
    
    def get_experiment(self, name: str) -> ExperimentConfig:
        """Get experiment configuration by name"""
        for exp in self._config['experiments']:
            if exp['name'] == name:
                return ExperimentConfig(
                    name=exp['name'],
                    notebook=exp['notebook'],
                    data_path=exp.get('data_path', ''),
                    description=exp.get('description', ''),
                    mlflow_experiment=exp.get('mlflow_experiment'),
                    hyperparameters=exp.get('hyperparameters', {}),
                    instance_type=exp.get('instance_type', 'ml.m5.xlarge'),
                    instance_count=exp.get('instance_count', 1)
                )
        
        raise ValueError(f"Experiment not found: {name}")
    
    def list_experiments(self) -> List[str]:
        """List all experiment names"""
        return [exp['name'] for exp in self._config['experiments']]
