# 🚀 Quick Start Guide

Get up and running with Notebooked in 5 minutes!

---

## Installation

```bash
pip install -e .
```

For cloud provider support:
```bash
pip install -e .[azure,gcp]  # Azure and GCP
```

---

## Your First Workflow

### Step 1: Tag Your Notebook

Open your Jupyter notebook and add tags to cells:

1. Click on a code cell
2. View → Cell Toolbar → Tags
3. Add appropriate tags:

**Example notebook structure:**

📓 **Cell 1** - Tagged: `imports`
```python
import torch
import torch.nn as nn
import mlflow
```

📓 **Cell 2** - Tagged: `preprocess`
```python
def load_data(data_path):
    # Your data loading code
    return train_data, val_data
```

📓 **Cell 3** - Tagged: `model`
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)
```

📓 **Cell 4** - Tagged: `train`
```python
def train_model(model, data, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        # Training loop
        loss = ...
        mlflow.log_metric("loss", loss, step=epoch)
    return model
```

📓 **Cell 5** - Tagged: `inference`
```python
def predict(model, data):
    return model(data)
```

### Step 2: Create Configuration

```bash
notebooked init
```

Edit `config.yaml`:
```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "my-first-model"

aws:
  region: "us-east-1"
  role: "arn:aws:iam::123456789012:role/SageMakerRole"

experiments:
  - name: "my-experiment"
    notebook: "notebooks/my_model.ipynb"
    data_path: "s3://my-bucket/data"
    hyperparameters:
      epochs: 10
      learning_rate: 0.001
```

### Step 3: Convert Notebook

```bash
notebooked convert my-experiment
```

Check the generated files:
```
generated/my-experiment/
├── train.py           # ✅ Training script with MLFlow
├── inference.py       # ✅ SageMaker inference handlers
├── preprocess.py      # ✅ Data preprocessing
└── requirements.txt   # ✅ Auto-detected dependencies
```

### Step 4: Train Locally (Test)

```bash
notebooked train my-experiment --provider local
```

### Step 5: Train on Cloud

**AWS SageMaker:**
```bash
notebooked train my-experiment --provider sagemaker --wait
```

**Azure ML:**
```bash
notebooked train my-experiment --provider azure --wait
```

**Google Cloud Vertex AI:**
```bash
notebooked train my-experiment --provider gcp --wait
```

### Step 6: Deploy Model

```bash
notebooked deploy my-experiment \
  --model-uri s3://my-bucket/model/model.tar.gz \
  --endpoint-name my-model-endpoint \
  --provider sagemaker
```

---

## Common Use Cases

### Case 1: Rapid Prototyping → Production

```bash
# 1. Experiment in notebook
# 2. Tag cells as you go
# 3. Convert when ready
notebooked convert my-prototype

# 4. Test locally
notebooked train my-prototype --provider local

# 5. Deploy to cloud
notebooked train my-prototype --provider sagemaker
```

### Case 2: Multi-Cloud Deployment

```bash
# Same notebook, multiple clouds
notebooked train my-model --provider sagemaker
notebooked train my-model --provider azure
notebooked train my-model --provider gcp
```

### Case 3: CI/CD Pipeline

```bash
# Generate workflow
notebooked generate-workflow --provider sagemaker

# Commit to git
git add .github/workflows/notebooked-pipeline.yml
git commit -m "Add ML pipeline"
git push

# GitHub Actions will automatically:
# - Convert notebook
# - Train on SageMaker
# - Run tests
# - Deploy if tests pass
```

---

## Troubleshooting

### Issue: "Invalid tags found"

**Solution**: Check that you're using valid tags:
- ✅ `imports`, `preprocess`, `model`, `train`, `inference`, `utils`, `requirements`
- ❌ `data`, `setup`, `config` (not valid)

### Issue: "Model class not found in generated script"

**Solution**: Make sure model definition has the `model` tag, not `train`.

### Issue: "Duplicate imports in generated script"

**Solution**: Only tag cells with `imports` if you want explicit control. Otherwise, let auto-detection handle it.

### Issue: "Training fails with import error"

**Solution**: Check `generated/*/requirements.txt` and add any missing dependencies:
```txt
torch>=2.0.0
transformers>=4.30.0
```

---

## Next Steps

- 📖 Read the full [README.md](../README.md)
- 🏗️ Understand the [Architecture](ARCHITECTURE.md)
- 🤝 Learn how to [Contribute](../CONTRIBUTING.md)
- 💡 Check out [example notebooks](../notebooks/)

---

**Happy ML Engineering! 🎉**
