"""Local provider implementation"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import shutil

from .base import Provider


class LocalProvider(Provider):
    """Local implementation of the Provider interface"""
    
    def __init__(self):
        pass
            
    def train(
        self,
        experiment_name: str,
        source_dir: Path,
        data_path: str,
        hyperparameters: Dict[str, Any],
        instance_type: str = "local",
        instance_count: int = 1,
        wait: bool = True
    ) -> Dict[str, Any]:
        """Run training job locally via subprocess"""
        print("=" * 80)
        print("STARTING LOCAL TRAINING")
        print("=" * 80)
        
        # Prepare environment
        env = os.environ.copy()
        
        # Setup model directory
        model_dir = Path("models") / experiment_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct command
        # We assume train.py is the entry point
        script_path = source_dir / "train.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")
            
        cmd = [sys.executable, str(script_path)]
        
        # Add arguments
        cmd.extend(["--data-path", data_path])
        cmd.extend(["--model-dir", str(model_dir)])
        
        # Add hyperparameters as args if they match the script's expectations
        # Note: The generated script uses global constants for hyperparameters, 
        # but we can also pass them if the script was modified to accept them.
        # For now, we rely on the generated constants.
        
        print(f"Running: {' '.join(cmd)}")
        print(f"Logs will be streamed below...\n")
        
        try:
            # Run process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            if wait:
                with process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        print(line, end='')
                
                returncode = process.wait()
                
                if returncode == 0:
                    status = "Completed"
                    print(f"\n✅ Local training completed successfully.")
                else:
                    status = "Failed"
                    print(f"\n❌ Local training failed with exit code {returncode}.")
            else:
                status = "InProgress"
                print(f"\nTraining started in background (PID: {process.pid})")
                
        except Exception as e:
            print(f"\n❌ Error executing local training: {e}")
            status = "Error"
            
        return {
            'job_name': f"local-{experiment_name}",
            'status': status,
            'model_uri': str(model_dir),
            'instance_type': 'local'
        }

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
        Deploy model locally (Mock implementation for now)
        In a real scenario, this could start a Flask/FastAPI server.
        """
        print("=" * 80)
        print("DEPLOYING LOCAL ENDPOINT (MOCK)")
        print("=" * 80)
        
        print(f"Model: {model_uri}")
        print(f"Endpoint: {endpoint_name}")
        print("Note: Local serving is not yet fully implemented. This is a placeholder.")
        
        return {
            'endpoint_name': endpoint_name,
            'endpoint_url': 'http://localhost:8080/invocations',
            'status': 'InService'
        }

    def predict(
        self,
        endpoint_name: str,
        data: Any
    ) -> Any:
        """Run inference on local endpoint"""
        print(f"Predicting on local endpoint: {endpoint_name}")
        # TODO: Implement actual HTTP request to local server
        return {"result": "mock_prediction"}

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete local endpoint"""
        print(f"Stopping local endpoint: {endpoint_name}")
