import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from notebooked.providers.sagemaker import SageMakerProvider

class TestSageMakerDeployment(unittest.TestCase):
    
    def setUp(self):
        self.region = "us-east-1"
        self.role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"
        
    @patch('notebooked.providers.sagemaker.boto3.Session')
    @patch('notebooked.providers.sagemaker.sagemaker.Session')
    @patch('notebooked.providers.sagemaker.PyTorchModel')
    def test_deploy_realtime(self, mock_model_cls, mock_sm_session, mock_boto_session):
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        
        mock_predictor = MagicMock()
        mock_predictor.endpoint_name = "test-endpoint"
        mock_model_instance.deploy.return_value = mock_predictor
        
        # Initialize provider
        provider = SageMakerProvider(region=self.region, role=self.role)
        
        # Call deploy
        result = provider.deploy(
            model_uri="s3://bucket/model.tar.gz",
            endpoint_name="test-endpoint",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            serverless=False
        )
        
        # Verify PyTorchModel was created with correct args
        mock_model_cls.assert_called_once()
        _, kwargs = mock_model_cls.call_args
        self.assertEqual(kwargs['model_data'], "s3://bucket/model.tar.gz")
        self.assertEqual(kwargs['role'], self.role)
        self.assertEqual(kwargs['entry_point'], "inference.py")
        
        # Verify deploy was called with correct args
        mock_model_instance.deploy.assert_called_once_with(
            endpoint_name="test-endpoint",
            instance_type="ml.m5.xlarge",
            initial_instance_count=1
        )
        
        # Verify result
        self.assertEqual(result['endpoint_name'], "test-endpoint")
        self.assertEqual(result['status'], "InService")

    @patch('notebooked.providers.sagemaker.boto3.Session')
    @patch('notebooked.providers.sagemaker.sagemaker.Session')
    @patch('notebooked.providers.sagemaker.PyTorchModel')
    def test_deploy_serverless(self, mock_model_cls, mock_sm_session, mock_boto_session):
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance
        
        mock_predictor = MagicMock()
        mock_predictor.endpoint_name = "test-serverless-endpoint"
        mock_model_instance.deploy.return_value = mock_predictor
        
        # Initialize provider
        provider = SageMakerProvider(region=self.region, role=self.role)
        
        # Call deploy
        result = provider.deploy(
            model_uri="s3://bucket/model.tar.gz",
            endpoint_name="test-serverless-endpoint",
            instance_type="ml.m5.xlarge", # Should be ignored for serverless
            instance_count=1,
            serverless=True,
            serverless_memory=4096,
            serverless_concurrency=10
        )
        
        # Verify deploy was called with serverless config
        mock_model_instance.deploy.assert_called_once()
        _, kwargs = mock_model_instance.deploy.call_args
        self.assertEqual(kwargs['endpoint_name'], "test-serverless-endpoint")
        self.assertIn('serverless_inference_config', kwargs)
        
        serverless_config = kwargs['serverless_inference_config']
        self.assertEqual(serverless_config.memory_size_in_mb, 4096)
        self.assertEqual(serverless_config.max_concurrency, 10)

if __name__ == '__main__':
    unittest.main()
