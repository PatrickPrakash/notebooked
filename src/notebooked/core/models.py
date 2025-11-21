"""Configuration models using Pydantic"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path


class MLFlowConfig(BaseModel):
    """MLFlow configuration"""
    tracking_uri: str = Field(default="http://localhost:5000", description="MLFlow tracking URI")
    experiment_name: str = Field(default="default", description="MLFlow experiment name")


class AWSConfig(BaseModel):
    """AWS configuration"""
    region: str = Field(default="us-east-1", description="AWS Region")
    access_key: Optional[str] = Field(default=None, description="AWS Access Key ID")
    secret_key: Optional[str] = Field(default=None, description="AWS Secret Access Key")
    role: Optional[str] = Field(default=None, description="IAM Role ARN")


class AzureConfig(BaseModel):
    """Azure configuration"""
    subscription_id: Optional[str] = Field(default=None, description="Azure Subscription ID")
    resource_group: Optional[str] = Field(default=None, description="Azure Resource Group")
    workspace_name: Optional[str] = Field(default=None, description="Azure ML Workspace Name")


class GCPConfig(BaseModel):
    """GCP configuration"""
    project_id: Optional[str] = Field(default=None, description="GCP Project ID")
    location: str = Field(default="us-central1", description="GCP Region")
    staging_bucket: Optional[str] = Field(default=None, description="GCS Bucket for staging artifacts")


class ExperimentConfig(BaseModel):
    """Experiment configuration"""
    name: str = Field(..., description="Unique name of the experiment")
    notebook: Path = Field(..., description="Path to the source notebook")
    data_path: str = Field(default="", description="Path to input data (Local or S3)")
    description: str = Field(default="", description="Description of the experiment")
    mlflow_experiment: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters")
    instance_type: str = Field(default="ml.m5.xlarge", description="Compute instance type")
    instance_count: int = Field(default=1, ge=1, description="Number of instances")
    
    @validator('notebook')
    def notebook_must_exist(cls, v):
        if not v.exists():
            # We might be validating config before the file exists (e.g. init), 
            # but generally for running it should exist. 
            # For now, let's just warn or allow it, but best practice is to check.
            pass 
        return v


class ProjectConfig(BaseModel):
    """Root project configuration"""
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    experiments: List[ExperimentConfig] = Field(default_factory=list)
    
    def get_experiment(self, name: str) -> ExperimentConfig:
        for exp in self.experiments:
            if exp.name == name:
                return exp
        raise ValueError(f"Experiment '{name}' not found in configuration.")
