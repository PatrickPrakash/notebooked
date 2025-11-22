# Contributing to Notebooked

Thank you for your interest in contributing to Notebooked! This document provides guidelines and instructions for contributing to the project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Adding New Providers](#adding-new-providers)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect:

- **Respectful Communication**: Treat everyone with respect and professionalism
- **Constructive Feedback**: Provide helpful, actionable feedback
- **Collaboration**: Work together towards shared goals
- **Openness**: Be open to different perspectives and ideas

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or insulting remarks
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of Jupyter notebooks and ML workflows
- Familiarity with at least one cloud provider (AWS/Azure/GCP) for provider-specific work

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/notebooked.git
   cd notebooked
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/patrickprakash/notebooked.git
   ```

4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .[dev]

# Install optional cloud provider dependencies
pip install -e .[azure,gcp]
```

### Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

---

## Project Structure

```
notebooked/
├── src/notebooked/
│   ├── core/
│   │   ├── parser.py          # Notebook parsing logic
│   │   ├── generator.py       # Code generation logic
│   │   ├── models.py          # Configuration models
│   │   ├── config.py          # Config loader
│   │   └── workflow.py        # CI/CD generation
│   ├── providers/
│   │   ├── base.py            # Abstract provider interface
│   │   ├── local.py           # Local execution
│   │   ├── sagemaker.py       # AWS SageMaker
│   │   ├── azure_ml.py        # Azure ML
│   │   └── vertex_ai.py       # Google Cloud Vertex AI
│   └── cli.py                 # CLI entry point
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test fixtures
├── notebooks/                 # Example notebooks
├── docs/                      # Documentation
└── pyproject.toml             # Project configuration
```

### Key Components

#### Core Modules

1. **parser.py**: Parses Jupyter notebooks and extracts tagged cells
   - `NotebookParser` class
   - `ExtractedCode` dataclass
   - Tag validation logic

2. **generator.py**: Generates Python scripts from extracted code
   - `CodeGenerator` class
   - Template-based generation
   - Requirements detection with pipreqs

3. **models.py**: Pydantic models for configuration
   - `ProjectConfig`
   - `AWSConfig`
   - `AzureConfig`
   - `GCPConfig`

#### Provider Modules

All providers inherit from `BaseProvider` in `providers/base.py`:

```python
class BaseProvider(ABC):
    @abstractmethod
    def train(self, experiment_name: str, wait: bool = True) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def deploy(self, experiment_name: str, model_uri: str, 
               endpoint_name: str, **kwargs) -> str:
        pass
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized (stdlib, third-party, local)

### Code Quality Tools

We use the following tools to maintain code quality:

1. **Ruff**: Fast Python linter
   ```bash
   ruff check src/
   ruff format src/  # Auto-format
   ```

2. **Type Hints**: Use type hints where applicable
   ```python
   def parse_notebook(path: str) -> ExtractedCode:
       ...
   ```

3. **Docstrings**: Use Google-style docstrings
   ```python
   def train(self, experiment_name: str, wait: bool = True) -> Dict[str, Any]:
       """Train a model using this provider.
       
       Args:
           experiment_name: Name of the experiment to train
           wait: Whether to wait for training to complete
           
       Returns:
           Dictionary containing training job information
           
       Raises:
           ValueError: If experiment not found in config
       """
   ```

### Naming Conventions

- **Classes**: PascalCase (`NotebookParser`, `SageMakerProvider`)
- **Functions/Methods**: snake_case (`parse_notebook`, `train_model`)
- **Constants**: UPPER_SNAKE_CASE (`VALID_TAGS`, `DEFAULT_REGION`)
- **Private Methods**: Leading underscore (`_filter_magic_commands`)

---

## Testing Guidelines

### Writing Tests

We use `pytest` for testing. All tests should be in the `tests/` directory.

#### Unit Tests

```python
# tests/unit/test_parser.py
import pytest
from notebooked.core.parser import NotebookParser

def test_parse_valid_notebook():
    parser = NotebookParser("tests/fixtures/sample.ipynb")
    parser.parse()
    extracted = parser.extract_tagged_code()
    
    assert len(extracted.train) > 0
    assert len(extracted.imports) > 0
```

#### Integration Tests

```python
# tests/integration/test_full_workflow.py
def test_convert_and_train_locally(tmp_path):
    # Full end-to-end test
    config = create_test_config(tmp_path)
    
    # Convert
    generator = CodeGenerator(output_dir=tmp_path)
    files = generator.generate_all(...)
    
    # Train
    provider = LocalProvider()
    result = provider.train("test-exp")
    
    assert result["status"] == "completed"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_parser.py

# Run with coverage
pytest --cov=src/notebooked --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run with verbose output
pytest -v
```

### Testing Best Practices

1. **Mock External Services**: Use `unittest.mock` or `pytest-mock` for cloud APIs
2. **Use Fixtures**: Create reusable test data in `conftest.py`
3. **Test Edge Cases**: Include tests for error conditions
4. **Keep Tests Fast**: Unit tests should run in milliseconds
5. **Descriptive Names**: Test names should clearly describe what they test

---

## Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass
   ```bash
   pytest
   ```

2. **Run Linting**: Fix any linting issues
   ```bash
   ruff check src/ --fix
   ruff format src/
   ```

3. **Update Documentation**: Update README or docs if needed

4. **Add Tests**: Include tests for new functionality

5. **Update Changelog**: Add entry to CHANGELOG.md (if exists)

### PR Guidelines

#### Title Format

Use conventional commits format:

- `feat: Add support for GCP Vertex AI deployment`
- `fix: Resolve duplicate imports in generated code`
- `docs: Update README with new examples`
- `test: Add integration tests for Azure provider`
- `refactor: Simplify generator template logic`

#### Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why are these changes needed?

## Changes Made
- List of specific changes
- Another change
- etc.

## Testing
How were these changes tested?

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Linting passes
- [ ] All tests pass
```

### Review Process

1. At least one maintainer must approve the PR
2. All CI checks must pass
3. No unresolved conversations
4. Code follows project standards

---

## Adding New Providers

To add support for a new cloud provider:

### 1. Create Provider Class

Create `src/notebooked/providers/your_provider.py`:

```python
from .base import BaseProvider
from typing import Dict, Any

class YourProvider(BaseProvider):
    def __init__(self, **config):
        self.config = config
    
    def train(self, experiment_name: str, wait: bool = True) -> Dict[str, Any]:
        """Implement training logic"""
        # Your implementation
        pass
    
    def deploy(self, experiment_name: str, model_uri: str, 
               endpoint_name: str, **kwargs) -> str:
        """Implement deployment logic"""
        # Your implementation
        pass
    
    def predict(self, endpoint_name: str, data: Any) -> Any:
        """Implement prediction logic"""
        # Your implementation
        pass
```

### 2. Add Configuration Model

Update `src/notebooked/core/models.py`:

```python
class YourProviderConfig(BaseModel):
    api_key: str
    region: str
    # Add provider-specific fields

class ProjectConfig(BaseModel):
    # ...existing fields...
    your_provider: Optional[YourProviderConfig] = None
```

### 3. Update CLI

Update `src/notebooked/cli.py` to handle the new provider:

```python
@main.command()
@click.option('--provider', type=click.Choice(['local', 'sagemaker', 'azure', 'gcp', 'your-provider']))
def train(ctx, experiment_name, provider, wait):
    # ...
    elif provider == 'your-provider':
        from .providers.your_provider import YourProvider
        impl = YourProvider(**config.your_provider.dict())
```

### 4. Add Workflow Template

Update `src/notebooked/core/workflow.py`:

```python
def _get_your_provider_template(self, branch: str) -> str:
    return f"""name: Notebooked Pipeline (Your Provider)
# ... workflow content ...
"""
```

### 5. Documentation

- Add provider docs to README.md
- Include configuration examples
- Add usage examples

### 6. Tests

Create `tests/unit/test_your_provider.py`:

```python
def test_your_provider_train():
    provider = YourProvider(api_key="test")
    result = provider.train("test-exp")
    assert result is not None
```

---

## Documentation

### Documentation Standards

1. **README.md**: User-facing documentation
2. **CONTRIBUTING.md**: This file
3. **Docstrings**: In-code documentation
4. **Type Hints**: Self-documenting code
5. **Examples**: In `notebooks/` directory

### Writing Good Documentation

- **Be Clear**: Use simple, direct language
- **Be Concise**: Avoid unnecessary words
- **Use Examples**: Show, don't just tell
- **Keep Updated**: Update docs with code changes
- **Use Diagrams**: Mermaid diagrams for complex flows

---

## Questions?

- **General Questions**: Open a [GitHub Discussion](https://github.com/patrickprakash/notebooked/discussions)
- **Bug Reports**: Open an [Issue](https://github.com/patrickprakash/notebooked/issues)
- **Feature Requests**: Open an [Issue](https://github.com/patrickprakash/notebooked/issues) with `enhancement` label

---

## Attribution

By contributing to Notebooked, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Notebooked! 🎉**
