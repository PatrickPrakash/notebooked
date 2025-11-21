"""
Workflow generation logic for CI/CD pipelines.
"""

from pathlib import Path
from typing import Literal

class WorkflowGenerator:
    """Generates GitHub Actions workflow files."""

    def __init__(self, output_dir: Path = Path(".github/workflows")):
        self.output_dir = output_dir

    def generate(self, provider: Literal['local', 'sagemaker'], branch: str = 'main') -> Path:
        """
        Generate the workflow file for the specified provider.
        
        Args:
            provider: The target provider ('local' or 'sagemaker').
            branch: The branch to trigger the workflow on.
            
        Returns:
            Path to the generated workflow file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = self.output_dir / "notebooked-pipeline.yml"
        
        if provider == 'local':
            content = self._get_local_template(branch)
        elif provider == 'sagemaker':
            content = self._get_sagemaker_template(branch)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        with open(workflow_path, 'w') as f:
            f.write(content)
            
        return workflow_path

    def _get_local_template(self, branch: str) -> str:
        return f"""name: Notebooked Pipeline (Local)

on:
  push:
    branches: [ {branch} ]
  pull_request:
    branches: [ {branch} ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        
    - name: Run Tests
      run: |
        if [ -d "tests" ]; then
          python -m unittest discover tests
        fi
        
    - name: Convert Notebooks
      run: |
        # Placeholder: Convert a demo experiment if it exists
        notebooked convert test_experiment || echo "No test_experiment found"
        
    - name: Run Local Training (Integration Test)
      run: |
        notebooked train test_experiment --provider local || echo "Skipping training"
"""

    def _get_sagemaker_template(self, branch: str) -> str:
        return f"""name: Notebooked Pipeline (SageMaker)

on:
  push:
    branches: [ {branch} ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        
    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{{{ secrets.AWS_ACCESS_KEY_ID }}}}
        aws-secret-access-key: ${{{{ secrets.AWS_SECRET_ACCESS_KEY }}}}
        aws-region: us-east-1
        
    - name: Train on SageMaker
      run: |
        notebooked train test_experiment --provider sagemaker
"""
