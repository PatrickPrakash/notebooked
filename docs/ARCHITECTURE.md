# Notebooked Architecture

This document provides a detailed technical overview of the Notebooked architecture, design patterns, and implementation details.

---

## System Architecture

### Component Diagram

```mermaid
graph TB
    subgraph "CLI Layer"
        CLI[CLI Commands<br/>click-based]
    end
    
    subgraph "Core Layer"
        PARSER[NotebookParser<br/>AST-based extraction]
        GEN[CodeGenerator<br/>Template engine]
        CONFIG[ConfigLoader<br/>Pydantic models]
        WORKFLOW[WorkflowGenerator<br/>CI/CD templates]
    end
    
    subgraph "Provider Layer"
        BASE[BaseProvider<br/>Abstract interface]
        LOCAL[LocalProvider]
        SM[SageMakerProvider]
        AZ[AzureProvider]
        GCP[VertexAIProvider]
    end
    
    subgraph "External Services"
        MLFLOW[MLFlow<br/>Tracking Server]
        AWS[AWS SageMaker]
        AZURE[Azure ML]
        GOOGLE[Google Cloud<br/>Vertex AI]
    end
    
    CLI --> PARSER
    CLI --> GEN
    CLI --> CONFIG
    CLI --> WORKFLOW
    CLI --> BASE
    
    BASE --> LOCAL
    BASE --> SM
    BASE --> AZ
    BASE --> GCP
    
    GEN --> MLFLOW
    SM --> AWS
    AZ --> AZURE
    GCP --> GOOGLE
```

---

## Data Flow

### Notebook Conversion Flow

```mermaid
flowchart TD
    START([User: notebooked convert]) --> LOAD[Load config.yaml]
    LOAD --> FIND[Find notebook path]
    FIND --> PARSE[Parse Jupyter notebook JSON]
    
    PARSE --> EXTRACT[Extract cells by tags]
    EXTRACT --> IMPORTS[Collect imports]
    EXTRACT --> PREPROC[Collect preprocess]
    EXTRACT --> MODEL[Collect model]
    EXTRACT --> TRAIN[Collect train]
    EXTRACT --> INFER[Collect inference]
    
    IMPORTS --> COMBINE[Combine ExtractedCode]
    PREPROC --> COMBINE
    MODEL --> COMBINE
    TRAIN --> COMBINE
    INFER --> COMBINE
    
    COMBINE --> GENTRAIN[Generate train.py]
    COMBINE --> GENINFER[Generate inference.py]
    COMBINE --> GENPREP[Generate preprocess.py]
    COMBINE --> GENREQ[Generate requirements.txt]
    
    GENTRAIN --> SAVE[Save to generated/]
    GENINFER --> SAVE
    GENPREP --> SAVE
    GENREQ --> SAVE
    
    SAVE --> END([✓ Success])
```

### Training Flow

```mermaid
flowchart TD
    START([User: notebooked train]) --> LOADCONF[Load config.yaml]
    LOADCONF --> GETEXP[Get experiment config]
    GETEXP --> CHECKSCRIPTS{Scripts exist?}
    
    CHECKSCRIPTS -->|No| CONVERT[Auto-convert notebook]
    CHECKSCRIPTS -->|Yes| PROVIDER
    CONVERT --> PROVIDER[Initialize provider]
    
    PROVIDER --> LOCAL{Local provider?}
    LOCAL -->|Yes| SUBPROCESS[Run train.py<br/>in subprocess]
    LOCAL -->|No| CLOUD[Submit cloud job]
    
    SUBPROCESS --> STREAM[Stream output]
    STREAM --> WAIT{--wait flag?}
    
    CLOUD --> UPLOAD[Upload code to cloud]
    UPLOAD --> SUBMIT[Submit training job]
    SUBMIT --> WAIT
    
    WAIT -->|Yes| POLL[Poll job status]
    WAIT -->|No| ASYNC[Return job ID]
    
    POLL --> CHECK{Job complete?}
    CHECK -->|No| POLL
    CHECK -->|Yes| SUCCESS
    ASYNC --> SUCCESS
    
    SUCCESS([✓ Training complete])
```

---

## Core Components

### 1. NotebookParser

**Purpose**: Extract code from tagged Jupyter notebook cells

**Key Features**:
- AST-based import detection
- Tag validation
- Magic command filtering
- Multi-cell code combination

**Algorithm**:

```mermaid
flowchart LR
    INPUT[Notebook JSON] --> LOAD[Load cells]
    LOAD --> ITERATE[For each cell]
    ITERATE --> TAG{Has tags?}
    
    TAG -->|Yes| VALID{Valid tag?}
    TAG -->|No| AUTOIMPORT{Auto-detect imports?}
    
    VALID -->|Yes| EXTRACT[Extract to category]
    VALID -->|No| WARN[Log warning]
    
    AUTOIMPORT -->|Yes| AST[Parse with AST]
    AUTOIMPORT -->|No| SKIP[Skip cell]
    
    AST --> EXTRACTIMPORT[Extract imports]
    EXTRACT --> COMBINE[Combine by category]
    EXTRACTIMPORT --> COMBINE
    
    COMBINE --> OUTPUT[ExtractedCode]
```

**Code Structure**:
```python
class NotebookParser:
    VALID_TAGS = {'imports', 'preprocess', 'model', 'train', 'inference', 'utils'}
    
    def parse() -> None:
        # Load notebook JSON
        # Parse cells
        
    def extract_tagged_code() -> ExtractedCode:
        # Auto-detect imports
        # Extract tagged cells
        # Filter magic commands
        # Combine code by tag
        
    def _auto_detect_imports():
        # Use AST to find import statements
        # Skip tagged cells
        
    def _filter_magic_commands():
        # Remove % and ! commands
```

### 2. CodeGenerator

**Purpose**: Generate production-ready Python scripts from extracted code

**Templates**:

```mermaid
graph LR
    EXTRACTED[ExtractedCode] --> TRAIN[train.py template]
    EXTRACTED --> INFER[inference.py template]
    EXTRACTED --> PREP[preprocess.py template]
    
    TRAIN --> INJECT1[Inject: imports, hyperparams,<br/>preprocess, model, train]
    INFER --> INJECT2[Inject: imports, model,<br/>inference, IO handlers]
    PREP --> INJECT3[Inject: imports,<br/>preprocess, utils]
    
    INJECT1 --> DISABLE[Disable user __main__]
    DISABLE --> OUTPUT1[Generated train.py]
    INJECT2 --> OUTPUT2[Generated inference.py]
    INJECT3 --> OUTPUT3[Generated preprocess.py]
```

**Key Features**:
- Template-based generation
- MLFlow integration injection
- Hyperparameter configuration
- User main block disabling
- CLI argument parsing generation

**Generated Script Structure**:

```python
# train.py structure
"""
1. Header docstring
2. Standard imports (os, argparse)
3. MLFlow imports
4. User imports (from 'imports' tag)
5. Hyperparameters (from config)
6. MLFlow setup function
7. Utility functions (from 'utils' tag)
8. Preprocessing functions (from 'preprocess' tag)
9. Model classes (from 'model' tag)
10. MLFlow helper functions
11. Training code (from 'train' tag)
12. Generated main() function
13. if __name__ == "__main__": main()
"""
```

### 3. Provider System

**Design Pattern**: Strategy Pattern with Abstract Base Class

```mermaid
classDiagram
    class BaseProvider {
        <<abstract>>
        +train(experiment_name, wait)
        +deploy(experiment_name, model_uri, endpoint_name)
        +predict(endpoint_name, data)
        +delete_endpoint(endpoint_name)
    }
    
    class LocalProvider {
        +train() subprocess execution
        +deploy() local model loading
        +predict() direct inference
    }
    
    class SageMakerProvider {
        -sagemaker_client
        -s3_client
        +train() submit training job
        +deploy() create endpoint
        +predict() invoke endpoint
    }
    
    class AzureProvider {
        -ml_client
        +train() submit command job
        +deploy() create online endpoint
        +predict() invoke endpoint
    }
    
    class VertexAIProvider {
        -aiplatform
        +train() create custom job
        +deploy() upload & deploy model
        +predict() predict via endpoint
    }
    
    BaseProvider <|-- LocalProvider
    BaseProvider <|-- SageMakerProvider
    BaseProvider <|-- AzureProvider
    BaseProvider <|-- VertexAIProvider
```

**Provider Selection**:

```mermaid
flowchart TD
    START[CLI --provider flag] --> SWITCH{Provider?}
    
    SWITCH -->|local| LOCAL[LocalProvider]
    SWITCH -->|sagemaker| SM[SageMakerProvider]
    SWITCH -->|azure| AZ[AzureProvider]
    SWITCH -->|gcp| GCP[VertexAIProvider]
    
    LOCAL --> INIT1[Initialize with no config]
    SM --> INIT2[Initialize with AWS config]
    AZ --> INIT3[Initialize with Azure config]
    GCP --> INIT4[Initialize with GCP config]
    
    INIT1 --> EXEC[Execute train/deploy]
    INIT2 --> EXEC
    INIT3 --> EXEC
    INIT4 --> EXEC
```

---

## Configuration Management

### Configuration Hierarchy

```mermaid
graph TB
    FILE[config.yaml] --> LOAD[Load YAML]
    LOAD --> VALIDATE[Pydantic validation]
    
    VALIDATE --> PROJECT[ProjectConfig]
    PROJECT --> MLFLOW[MLFlowConfig]
    PROJECT --> AWS[AWSConfig]
    PROJECT --> AZURE[AzureConfig]
    PROJECT --> GCP[GCPConfig]
    PROJECT --> EXPS[List of ExperimentConfig]
    
    EXPS --> EXP1[Experiment 1]
    EXPS --> EXP2[Experiment 2]
    
    EXP1 --> NAME[name]
    EXP1 --> NB[notebook path]
    EXP1 --> DATA[data_path]
    EXP1 --> HYPER[hyperparameters]
```

**Validation Flow**:

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant ConfigLoader
    participant Pydantic
    
    User->>CLI: notebooked train exp1
    CLI->>ConfigLoader: load_config()
    ConfigLoader->>Pydantic: Parse YAML
    Pydantic->>Pydantic: Validate types
    Pydantic->>Pydantic: Check required fields
    
    alt Valid
        Pydantic-->>ConfigLoader: ProjectConfig
        ConfigLoader-->>CLI: config
        CLI->>CLI: Proceed with training
    else Invalid
        Pydantic-->>ConfigLoader: ValidationError
        ConfigLoader-->>CLI: Error
        CLI-->>User: ❌ Config error
    end
```

---

## Workflow Generation

### CI/CD Template Selection

```mermaid
flowchart TD
    CMD[notebooked generate-workflow] --> PROVIDER{--provider?}
    
    PROVIDER -->|local| LOCAL[Local template]
    PROVIDER -->|sagemaker| SM[SageMaker template]
    PROVIDER -->|azure| AZ[Azure template]
    PROVIDER -->|gcp| GCP[GCP template]
    
    LOCAL --> STRUCT1[GitHub Actions<br/>+ Python setup<br/>+ pip install<br/>+ run tests]
    
    SM --> STRUCT2[GitHub Actions<br/>+ AWS credentials<br/>+ notebooked train]
    
    AZ --> STRUCT3[GitHub Actions<br/>+ Azure login<br/>+ notebooked train]
    
    GCP --> STRUCT4[GitHub Actions<br/>+ GCP auth<br/>+ notebooked train]
    
    STRUCT1 --> WRITE[Write .github/workflows/]
    STRUCT2 --> WRITE
    STRUCT3 --> WRITE
    STRUCT4 --> WRITE
```

---

## Error Handling Strategy

### Error Propagation

```mermaid
flowchart TD
    ERROR[Error occurs] --> TYPE{Error type?}
    
    TYPE -->|FileNotFoundError| FILE[Specific message]
    TYPE -->|ValidationError| VALID[Config issue]
    TYPE -->|APIError| API[Cloud provider]
    TYPE -->|Other| GENERIC[Generic handler]
    
    FILE --> LOG[Log error]
    VALID --> LOG
    API --> LOG
    GENERIC --> LOG
    
    LOG --> USER[Display to user]
    USER --> EXIT[sys.exit(1)]
```

**Error Categories**:

1. **User Errors** (exit code 1):
   - Invalid configuration
   - Missing files
   - Invalid tags
   
2. **System Errors** (exit code 2):
   - Cloud provider API failures
   - Network issues
   
3. **Internal Errors** (exit code 3):
   - Unexpected exceptions
   - Logic errors

---

## Performance Considerations

### Optimization Strategies

1. **Lazy Loading**: Providers only imported when needed
2. **Caching**: Config loaded once per command
3. **Async Cloud Operations**: Optional `--no-wait` flag
4. **Minimal Dependencies**: Core has no cloud SDK dependencies

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Parse notebook | ~100ms | For 50-cell notebook |
| Generate scripts | ~50ms | 3 scripts + requirements |
| Submit SageMaker job | ~2s | Network dependent |
| Local training | Varies | Depends on model |

---

## Security Considerations

### Secrets Management

```mermaid
flowchart LR
    SECRETS[Cloud Credentials] --> ENV[Environment Variables]
    SECRETS --> GH[GitHub Secrets]
    SECRETS --> CONF[Config files<br/>gitignored]
    
    ENV --> PROVIDER[Provider SDK]
    GH --> ACTIONS[GitHub Actions]
    CONF --> LOCAL[Local execution]
```

**Best Practices**:
1. Never commit `config.yaml` with real credentials
2. Use environment variables for CI/CD
3. Use cloud provider's IAM roles when possible
4. Rotate credentials regularly

---

## Extension Points

### Adding Custom Features

1. **Custom Tags**: Modify `NotebookParser.VALID_TAGS`
2. **Custom Templates**: Extend `CodeGenerator` methods
3. **Custom Providers**: Inherit from `BaseProvider`
4. **Custom Workflows**: Add templates to `WorkflowGenerator`

---

## Future Enhancements

### Planned Features

```mermaid
mindmap
  root((Notebooked Future))
    Multi-Model
      A/B Testing
      Model Versioning
      Rollback Support
    Advanced Providers
      Databricks
      Kubeflow
      Custom K8s
    Enhanced Features
      Auto-hyperparameter tuning
      Model monitoring
      Data versioning
    UI/UX
      Web dashboard
      VS Code extension
      Jupyter extension
```

---

## Debugging Guide

### Debug Mode

Enable verbose logging:
```bash
export NOTEBOOKED_DEBUG=1
notebooked train my-exp
```

### Common Issues

1. **Duplicate Imports**: Check tag configuration
2. **Missing Model Classes**: Ensure 'model' tag is used
3. **Provider Errors**: Verify cloud credentials
4. **Generation Failures**: Check notebook structure

---

This architecture is designed to be:
- **Modular**: Each component has a single responsibility
- **Extensible**: Easy to add new providers and features
- **Testable**: Clear interfaces for mocking
- **Maintainable**: Consistent patterns throughout
