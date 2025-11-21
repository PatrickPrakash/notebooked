"""Azure Machine Learning provider implementation"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from azure.ai.ml import MLClient, command, Input
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        Model,
        Environment,
        CodeConfiguration
    )
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.constants import AssetTypes
except ImportError:
    MLClient = None

from .base import Provider


class AzureProvider(Provider):
    """Azure Machine Learning implementation of the Provider interface"""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str
    ):
        if MLClient is None:
            raise ImportError("azure-ai-ml is required for AzureProvider. Install with 'pip install notebooked[azure]'")
            
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        # Initialize MLClient
        # We assume DefaultAzureCredential works (env vars, az login, etc.)
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id,
            resource_group,
            workspace_name
        )

    def train(
        self,
        experiment_name: str,
        source_dir: Path,
        data_path: str,
        hyperparameters: Dict[str, Any],
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1,
        wait: bool = True
    ) -> Dict[str, Any]:
        """Submit a training job to Azure ML"""
        print(f"Submitting training job to Azure ML workspace: {self.workspace_name}")
        
        # Define the command
        # We assume data_path is a registered data asset or a URI
        # For simplicity, we pass it as an argument
        
        command_str = "python train.py --data-path ${{inputs.data}}"
        
        # Create the job
        job = command(
            code=str(source_dir),
            command=command_str,
            inputs={
                "data": Input(
                    type=AssetTypes.URI_FOLDER, 
                    path=data_path
                )
            },
            environment="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest", # Default environment
            compute=instance_type if instance_type != "local" else "local",
            display_name=experiment_name,
            experiment_name=experiment_name,
            instance_count=instance_count,
        )
        
        # Submit
        returned_job = self.ml_client.jobs.create_or_update(job)
        print(f"Job submitted. Studio URL: {returned_job.studio_url}")
        
        if wait:
            print("Waiting for job completion...")
            self.ml_client.jobs.stream(returned_job.name)
            
        return {
            'job_name': returned_job.name,
            'status': returned_job.status,
            'model_uri': f"azureml://jobs/{returned_job.name}/outputs/artifacts/paths/model/", # Approximate URI
            'studio_url': returned_job.studio_url
        }

    def deploy(
        self,
        model_uri: str,
        endpoint_name: str,
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1,
        serverless: bool = False, # Azure doesn't have "serverless" in the same way as SageMaker, but has managed endpoints
        serverless_memory: int = 2048,
        serverless_concurrency: int = 5
    ) -> Dict[str, Any]:
        """Deploy to Azure Managed Online Endpoint"""
        print(f"Deploying to Azure Endpoint: {endpoint_name}")
        
        # 1. Create Endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Deployed via notebooked",
            auth_mode="key"
        )
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        # 2. Register Model (if not already registered)
        # For simplicity, we assume model_uri points to a path we can register
        model = Model(
            path=model_uri,
            name=f"{endpoint_name}-model",
            type=AssetTypes.CUSTOM_MODEL,
            description="Model for deployment"
        )
        registered_model = self.ml_client.models.create_or_update(model)
        
        # 3. Create Deployment
        deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=registered_model,
            instance_type=instance_type,
            instance_count=instance_count,
        )
        self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        # 4. Update traffic
        endpoint.traffic = {"blue": 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        return {
            'endpoint_name': endpoint_name,
            'status': 'InService',
            'scoring_uri': endpoint.scoring_uri
        }

    def predict(self, endpoint_name: str, data: Any) -> Any:
        """Invoke endpoint"""
        # This typically requires a separate client or HTTP request
        # For brevity, we'll just print instructions
        print(f"To predict, send a POST request to the scoring URI of endpoint {endpoint_name}")
        return {}

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete endpoint"""
        print(f"Deleting endpoint: {endpoint_name}")
        self.ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
