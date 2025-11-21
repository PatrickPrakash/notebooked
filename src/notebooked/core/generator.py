"""Code generator to create Python scripts from extracted notebook code"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from .parser import ExtractedCode


class CodeGenerator:
    """Generate Python scripts from extracted notebook code"""
    
    def __init__(self, output_dir: str = "generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _use_pipreqs(self, output_dir: Path) -> bool:
        """Try to use pipreqs to generate requirements.txt"""
        try:
            # Check if pipreqs is available
            try:
                import pipreqs.pipreqs as pipreqs_module
                
                # Use pipreqs as a library
                print("   Using pipreqs to generate requirements...")
                
                # Prepare arguments in docopt format (as expected by pipreqs.init)
                args = {
                    '<path>': str(output_dir),
                    '--savepath': str(output_dir / 'requirements.txt'),
                    '--use-local': False,
                    '--force': True,
                    '--scan-notebooks': False,
                    '--print': False,
                    '--mode': 'compat',  # Use ~= for version pinning
                    '--ignore': None,
                    '--encoding': 'utf-8',
                    '--pypi-server': None,
                    '--proxy': None,
                    '--no-follow-links': False,
                    '--debug': False,
                    '--diff': None,
                    '--clean': None,
                    '--ignore-errors': False,
                }
                
                # Run pipreqs
                pipreqs_module.init(args)
                
                # Add core ML packages that might not be detected
                self._append_core_packages(output_dir / 'requirements.txt')
                
                print("   ✓ Requirements generated using pipreqs")
                return True
                
            except ImportError:
                # Try using pipreqs as subprocess
                print("   Trying pipreqs via command line...")
                result = subprocess.run(
                    ['pipreqs', str(output_dir), '--force', '--mode', 'compat'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    self._append_core_packages(output_dir / 'requirements.txt')
                    print("   ✓ Requirements generated using pipreqs CLI")
                    return True
                else:
                    print(f"   pipreqs CLI failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            print(f"   pipreqs not available or failed: {e}")
            return False
    
    def _append_core_packages(self, requirements_file: Path) -> None:
        """Append core ML packages to requirements.txt if not present"""
        core_packages = {
            'mlflow': 'mlflow>=2.8.0',
            'boto3': 'boto3>=1.28.0',
            'sagemaker': 'sagemaker>=2.190.0',
        }
        
        # Read existing requirements
        existing_content = ""
        if requirements_file.exists():
            with open(requirements_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Check which core packages are missing
        missing_packages = []
        for package_name, package_spec in core_packages.items():
            if package_name not in existing_content.lower():
                missing_packages.append(package_spec)
        
        # Append missing packages
        if missing_packages:
            with open(requirements_file, 'a', encoding='utf-8') as f:
                f.write('\n# Core ML pipeline packages\n')
                f.write('\n'.join(missing_packages))
                f.write('\n')
    
    
    def generate_all(
        self, 
        extracted: ExtractedCode, 
        experiment_name: str,
        hyperparameters: Dict[str, Any],
        mlflow_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """Generate all Python scripts"""
        generated_files = {}
        
        # Create experiment directory
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate preprocess.py
        if extracted.preprocess:
            preprocess_path = self._generate_preprocess(
                extracted, exp_dir, hyperparameters
            )
            generated_files['preprocess'] = preprocess_path
        
        # Generate train.py (with MLFlow integration)
        if extracted.train:
            train_path = self._generate_train(
                extracted, exp_dir, hyperparameters, mlflow_config
            )
            generated_files['train'] = train_path
        
        # Generate inference.py
        if extracted.inference:
            inference_path = self._generate_inference(
                extracted, exp_dir, hyperparameters
            )
            generated_files['inference'] = inference_path
        
        # Generate requirements.txt
        requirements_path = self._generate_requirements(exp_dir, extracted)
        generated_files['requirements'] = requirements_path
        
        return generated_files
    
    def _generate_preprocess(
        self,
        extracted: ExtractedCode,
        output_dir: Path,
        hyperparameters: Dict[str, Any]
    ) -> Path:
        """Generate preprocessing script"""
        imports = extracted.get_combined_code('imports')
        preprocess_code = extracted.get_combined_code('preprocess')
        utils_code = extracted.get_combined_code('utils')
        
        script = f'''"""
Preprocessing Script - Auto-generated from notebook
"""

import os
import argparse
from pathlib import Path

{imports}


# Hyperparameters
{self._format_hyperparameters(hyperparameters)}


# Utility functions
{utils_code if utils_code else "# No utility functions"}


# Preprocessing code
{preprocess_code}


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--data-path', type=str, required=True, help='Path to input data')
    parser.add_argument('--output-path', type=str, default='preprocessed_data', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Loading data from: {{args.data_path}}")
    print(f"Output directory: {{args.output_path}}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run preprocessing
    # Note: Data scientists should ensure their preprocessing code
    # saves the output to args.output_path
    
    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
'''
        
        output_path = output_dir / "preprocess.py"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        return output_path
    
    def _generate_train(
        self,
        extracted: ExtractedCode,
        output_dir: Path,
        hyperparameters: Dict[str, Any],
        mlflow_config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Generate training script with MLFlow integration"""
        imports = extracted.get_combined_code('imports')
        preprocess_code = extracted.get_combined_code('preprocess')
        train_code = extracted.get_combined_code('train')
        utils_code = extracted.get_combined_code('utils')
        
        mlflow_setup = ""
        if mlflow_config:
            mlflow_setup = f'''
# MLFlow Configuration
MLFLOW_TRACKING_URI = "{mlflow_config.get('tracking_uri', 'http://localhost:30500')}"
MLFLOW_EXPERIMENT = "{mlflow_config.get('experiment_name', 'default')}"

def setup_mlflow():
    """Setup MLFlow tracking"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLFlow tracking URI: {{MLFLOW_TRACKING_URI}}")
    print(f"MLFlow experiment: {{MLFLOW_EXPERIMENT}}")
'''
        else:
            mlflow_setup = '''
def setup_mlflow():
    """Mock MLFlow setup"""
    pass
'''

        script = f'''"""
Training Script - Auto-generated from notebook
Includes MLFlow tracking integration
"""

import os
import argparse
from pathlib import Path
import mlflow
import mlflow.pytorch

{imports}


# Hyperparameters
{self._format_hyperparameters(hyperparameters)}

{mlflow_setup}

# Utility functions
{utils_code if utils_code else "# No utility functions"}


# Data preprocessing functions
{preprocess_code if preprocess_code else "# No preprocessing functions"}


def log_metrics(metrics: dict, step: int = None):
    """Log metrics to MLFlow"""
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def log_parameters(params: dict):
    """Log parameters to MLFlow"""
    mlflow.log_params(params)


# Training code
{train_code}


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model-dir', type=str, 
                       default=os.environ.get('SM_MODEL_DIR', 'model'),
                       help='Model output directory')
    
    # Add hyperparameter arguments dynamically if possible, or just use defaults
    # For now, we rely on the global constants defined above
    
    args, unknown = parser.parse_known_args()
    
    # Setup MLFlow
    setup_mlflow()
    
    # Start MLFlow run
    with mlflow.start_run():
        print("Training started...")
        
        print(f"Data path: {{args.data_path}}")
        print(f"Model directory: {{args.model_dir}}")
        
        # Create model directory
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Execute training workflow
        # This part expects the user's code to run the training loop
        # and save the model to args.model_dir
        
        print("Training completed!")


if __name__ == "__main__":
    main()
'''
        
        output_path = output_dir / "train.py"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        return output_path
    
    def _generate_inference(
        self,
        extracted: ExtractedCode,
        output_dir: Path,
        hyperparameters: Dict[str, Any]
    ) -> Path:
        """Generate inference script"""
        imports = extracted.get_combined_code('imports')
        preprocess_code = extracted.get_combined_code('preprocess')
        train_code = extracted.get_combined_code('train')  # Need model architecture
        inference_code = extracted.get_combined_code('inference')
        utils_code = extracted.get_combined_code('utils')
        
        script = f'''"""
Inference Script - Auto-generated from notebook
"""

import os
import argparse
from pathlib import Path
import torch
import json

{imports}


# Hyperparameters
{self._format_hyperparameters(hyperparameters)}


# Utility functions
{utils_code if utils_code else "# No utility functions"}


# Data preprocessing functions (for tokenization)
{preprocess_code if preprocess_code else "# No preprocessing functions"}


# Model architecture (needed to load model from state_dict)
{train_code if train_code else "# No model architecture"}


# Inference code
{inference_code}


def model_fn(model_dir):
    """
    Load the PyTorch model from the model directory.
    This is used by SageMaker for deployment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'best_model.pth')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {{model_dir}}")
        
    print(f"Loading model from {{model_path}}")
    model = torch.load(model_path, map_location=device)
    
    if hasattr(model, 'eval'):
        model.eval()
    
    return model


def predict_fn(input_data, model):
    """
    Make predictions on input data.
    This is used by SageMaker for deployment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        # Ensure input is on correct device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device)
        
        # Run inference
        output = model(input_data)
    
    return output


def input_fn(request_body, request_content_type):
    """
    Deserialize the request body
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        # Convert to tensor if needed, depending on model expectation
        # For now, return as is
        return data
    raise ValueError(f"Unsupported content type: {{request_content_type}}")


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result
    """
    if response_content_type == 'application/json':
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().tolist()
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {{response_content_type}}")


def main():
    """Main inference function for standalone testing"""
    parser = argparse.ArgumentParser(description='Model inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input data')
    parser.add_argument('--output-path', type=str, default='predictions.csv', help='Output file')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {{args.model_path}}")
    print(f"Input data: {{args.input_path}}")
    
    # Load model
    model = model_fn(os.path.dirname(args.model_path))
    
    print(f"Predictions saved to: {{args.output_path}}")


if __name__ == "__main__":
    main()
'''
        
        output_path = output_dir / "inference.py"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        return output_path
    
    def _generate_requirements(self, output_dir: Path, extracted: ExtractedCode = None) -> Path:
        """Generate requirements.txt using pipreqs"""
        
        output_path = output_dir / "requirements.txt"
        
        # Check if requirements were manually tagged
        if extracted and extracted.requirements:
            print("   Using tagged requirements...")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted.get_combined_code('requirements'))
            return output_path

        # Use pipreqs to generate requirements
        if self._use_pipreqs(output_dir):
            return output_path
        
        # Fallback: Create minimal requirements
        print("   ⚠️  Failed to generate requirements with pipreqs. Creating minimal requirements.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("torch\ntorchvision\nmlflow\nboto3\nsagemaker\n")
            
        return output_path
    
    def _format_hyperparameters(self, hyperparameters: Dict[str, Any]) -> str:
        """Format hyperparameters as Python constants"""
        if not hyperparameters:
            return "# No hyperparameters defined"
            
        lines = []
        for key, value in hyperparameters.items():
            const_name = key.upper().replace('-', '_')
            
            if isinstance(value, str):
                lines.append(f'{const_name} = "{value}"')
            else:
                lines.append(f'{const_name} = {value}')
        
        return '\n'.join(lines)
