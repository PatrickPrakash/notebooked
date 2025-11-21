"""Google Cloud Vertex AI provider implementation"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None

from .base import Provider


class VertexAIProvider(Provider):
    """Vertex AI implementation of the Provider interface"""
    
    def __init__(
        self,
        project_id: str,
        location: str,
        staging_bucket: str
    ):
        if aiplatform is None:
            raise ImportError("google-cloud-aiplatform is required for VertexAIProvider. Install with 'pip install notebooked[gcp]'")
            
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket
        
        # Initialize Vertex AI SDK
        aiplatform.init(
            project=project_id,
            location=location,
            staging_bucket=staging_bucket
        )

    def train(
        self,
        experiment_name: str,
        source_dir: Path,
        data_path: str,
        hyperparameters: Dict[str, Any],
        instance_type: str = "n1-standard-4",
        instance_count: int = 1,
        wait: bool = True
    ) -> Dict[str, Any]:
        """Submit a custom training job to Vertex AI"""
        print(f"Submitting training job to Vertex AI project: {self.project_id}")
        
        # Define the Custom Training Job
        # We assume a pre-built container for simplicity
        job = aiplatform.CustomTrainingJob(
            display_name=experiment_name,
            script_path=str(source_dir / "train.py"),
            container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
            requirements=["mlflow", "boto3"], # Add dependencies needed by the script
            model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
        )
        
        # Run the job
        # Note: data_path handling in Vertex AI often involves GCS URIs passed as args
        args = ["--data-path", data_path]
        
        if wait:
            model = job.run(
                args=args,
                replica_count=instance_count,
                machine_type=instance_type,
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                sync=True
            )
            status = "Completed"
            model_name = model.resource_name
        else:
            job.run(
                args=args,
                replica_count=instance_count,
                machine_type=instance_type,
                sync=False
            )
            status = "InProgress"
            model_name = "pending"
            
        return {
            'job_name': job.display_name,
            'status': status,
            'model_uri': model_name, # In Vertex AI, this is the Model resource name
        }

    def deploy(
        self,
        model_uri: str,
        endpoint_name: str,
        instance_type: str = "n1-standard-4",
        instance_count: int = 1,
        serverless: bool = False,
        serverless_memory: int = 2048,
        serverless_concurrency: int = 5
    ) -> Dict[str, Any]:
        """Deploy to Vertex AI Endpoint"""
        print(f"Deploying to Vertex AI Endpoint: {endpoint_name}")
        
        # 1. Get the model (assuming model_uri is the resource name or ID)
        model = aiplatform.Model(model_name=model_uri)
        
        # 2. Create Endpoint (or get existing)
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        
        # 3. Deploy
        model.deploy(
            endpoint=endpoint,
            machine_type=instance_type,
            min_replica_count=instance_count,
            max_replica_count=instance_count,
        )
        
        return {
            'endpoint_name': endpoint.resource_name,
            'status': 'InService',
            'scoring_uri': endpoint.resource_name # Vertex AI uses resource name for prediction
        }

    def predict(self, endpoint_name: str, data: Any) -> Any:
        """Invoke endpoint"""
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        prediction = endpoint.predict(instances=[data])
        return prediction

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete endpoint"""
        print(f"Deleting endpoint: {endpoint_name}")
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        endpoint.undeploy_all()
        endpoint.delete()
