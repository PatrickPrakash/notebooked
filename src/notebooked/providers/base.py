"""Base provider interface"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class Provider(ABC):
    """Abstract base class for compute providers"""
    
    @abstractmethod
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
        """
        Run training job
        
        Args:
            experiment_name: Name of the experiment
            source_dir: Directory containing generated scripts
            data_path: Path to input data (S3 URI or local path)
            hyperparameters: Dictionary of hyperparameters
            instance_type: Compute instance type
            instance_count: Number of instances
            wait: Whether to wait for completion
            
        Returns:
            Dictionary containing job details (job_name, status, model_uri, etc.)
        """
        pass
    
    @abstractmethod
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
        """
        Deploy model to endpoint
        
        Args:
            model_uri: URI of the model artifact
            endpoint_name: Name for the endpoint
            instance_type: Compute instance type (for real-time)
            instance_count: Number of instances (for real-time)
            serverless: Whether to use serverless inference
            serverless_memory: Memory for serverless endpoint
            serverless_concurrency: Concurrency for serverless endpoint
            
        Returns:
            Dictionary containing endpoint details
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        endpoint_name: str,
        data: Any
    ) -> Any:
        """
        Run inference on deployed endpoint
        
        Args:
            endpoint_name: Name of the endpoint
            data: Input data (JSON serializable)
            
        Returns:
            Prediction result
        """
        pass
    
    @abstractmethod
    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete an endpoint"""
        pass
