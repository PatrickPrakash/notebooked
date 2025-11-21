"""SageMaker provider implementation"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker import get_execution_role
from pathlib import Path
from typing import Dict, Any, Optional
import time
import os
import json

from .base import Provider


class SageMakerProvider(Provider):
    """SageMaker implementation of the Provider interface"""
    
    def __init__(
        self,
        region: str,
        role: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        self.region = region
        
        # Configure AWS session
        if access_key and secret_key:
            self.session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
        else:
            self.session = boto3.Session(region_name=region)
            
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)
        self.s3_client = self.session.client('s3')
        
        # Get or use provided IAM role
        try:
            self.role = role or get_execution_role(sagemaker_session=self.sagemaker_session)
        except:
            if role is None:
                raise ValueError(
                    "IAM role required when not running in SageMaker environment. "
                    "Please provide role parameter."
                )
            self.role = role
            
    def train(
        self,
        experiment_name: str,
        source_dir: Path,
        data_path: str,
        hyperparameters: Dict[str, Any],
        instance_type: str,
        instance_count: int,
        wait: bool = True
    ) -> Dict[str, Any]:
        """Run training job on SageMaker"""
        print("=" * 80)
        print("SUBMITTING SAGEMAKER TRAINING JOB")
        print("=" * 80)
        
        # Normalize hyperparameters
        sm_hyperparameters = self._normalize_hyperparameters(hyperparameters)
        
        # Add data path for SageMaker
        sm_hyperparameters["data-path"] = "/opt/ml/input/data/training"
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point="train.py",
            source_dir=str(source_dir),
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='2.0.0', # Updated to match recent versions
            py_version='py310',
            hyperparameters=sm_hyperparameters,
            sagemaker_session=self.sagemaker_session,
            base_job_name=experiment_name
        )
        
        job_name = f"{experiment_name}-{int(time.time())}"
        
        print(f"\nJob Name: {job_name}")
        print(f"Instance Type: {instance_type}")
        print(f"Data: {data_path}")
        
        estimator.fit(
            {'training': data_path},
            job_name=job_name,
            wait=wait
        )
        
        result = {
            'job_name': job_name,
            'status': 'InProgress' if not wait else 'Completed',
            'instance_type': instance_type,
            'model_uri': estimator.model_data if wait else None
        }
        
        return result

    def deploy(
        self,
        model_uri: str,
        endpoint_name: str,
        instance_type: str,
        instance_count: int,
        serverless: bool = False,
        serverless_memory: int = 2048,
        serverless_concurrency: int = 5
    ) -> Dict[str, Any]:
        """Deploy model to SageMaker endpoint"""
        print("=" * 80)
        print("DEPLOYING MODEL TO SAGEMAKER ENDPOINT")
        print("=" * 80)
        
        # Create PyTorch model
        model = PyTorchModel(
            model_data=model_uri,
            role=self.role,
            framework_version='2.0.0',
            py_version='py310',
            entry_point='inference.py',
            # We assume source_dir is packaged in model.tar.gz or we'd need to pass it here
            # For simplicity in this migration, we assume model_uri contains the code
            # If not, we might need to repackage or pass source_dir
            sagemaker_session=self.sagemaker_session
        )
        
        print(f"\nEndpoint Name: {endpoint_name}")
        
        if serverless:
            print(f"Mode: Serverless")
            print(f"Memory: {serverless_memory} MB")
            
            serverless_config = ServerlessInferenceConfig(
                memory_size_in_mb=serverless_memory,
                max_concurrency=serverless_concurrency
            )
            
            predictor = model.deploy(
                endpoint_name=endpoint_name,
                serverless_inference_config=serverless_config
            )
        else:
            print(f"Mode: Real-time")
            print(f"Instance Type: {instance_type}")
            
            predictor = model.deploy(
                endpoint_name=endpoint_name,
                instance_type=instance_type,
                initial_instance_count=instance_count
            )
            
        return {
            'endpoint_name': endpoint_name,
            'endpoint_url': predictor.endpoint_name,
            'status': 'InService'
        }

    def predict(
        self,
        endpoint_name: str,
        data: Any
    ) -> Any:
        """Run inference on deployed endpoint"""
        runtime_client = self.session.client('sagemaker-runtime')
        
        # Serialize payload
        if isinstance(data, (dict, list)):
            body = json.dumps(data)
            content_type = 'application/json'
        else:
            body = data
            content_type = 'application/json' # Assume JSON string
            
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=body
        )
        
        result = response['Body'].read()
        
        try:
            return json.loads(result)
        except:
            return result.decode('utf-8')

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete a SageMaker endpoint"""
        sm_client = self.session.client('sagemaker')
        try:
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Endpoint deleted: {endpoint_name}")
        except Exception as e:
            print(f"Error deleting endpoint: {e}")

    def _normalize_hyperparameters(self, hyperparameters: Dict[str, Any]) -> Dict[str, str]:
        """Normalize hyperparameters for SageMaker CLI"""
        normalized = {}
        for key, value in hyperparameters.items():
            # Convert to string as SageMaker expects string values
            # Convert key underscores to dashes
            cli_key = key.replace("_", "-")
            normalized[cli_key] = str(value)
        return normalized
