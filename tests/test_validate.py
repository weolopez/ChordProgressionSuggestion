#!/usr/bin/env python
"""
Test the model validation functionality.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from validate_model import check_model_structure, create_dummy_input, run_inference


class TestValidate(unittest.TestCase):
    """Test the model validation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path('./test_validate')
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            # Remove all files in the test directory
            for file in self.test_dir.glob('*'):
                if file.is_file():
                    file.unlink()
            
            # Remove the test directory
            self.test_dir.rmdir()
    
    def test_create_dummy_input(self):
        """Test creating dummy input data."""
        # Define test input metadata
        input_info = [
            {"name": "input1", "shape": [None, 10, 20]},
            {"name": "input2", "shape": [5, 15]}
        ]
        
        # Create dummy inputs
        batch_size = 3
        dummy_inputs = create_dummy_input(input_info, batch_size)
        
        # Verify the result
        self.assertIn("input1", dummy_inputs)
        self.assertIn("input2", dummy_inputs)
        
        # Check shapes
        self.assertEqual(dummy_inputs["input1"].shape, (batch_size, 10, 20))
        self.assertEqual(dummy_inputs["input2"].shape, (5, 15))
        
        # Check data types
        self.assertEqual(dummy_inputs["input1"].dtype, np.float32)
        self.assertEqual(dummy_inputs["input2"].dtype, np.float32)
    
    @mock.patch('onnx.load')
    @mock.patch('onnx.checker.check_model')
    def test_check_model_structure(self, mock_check, mock_load):
        """Test checking the structure of an ONNX model."""
        # Create mock objects
        mock_model = mock.MagicMock()
        mock_graph = mock.MagicMock()
        mock_model.graph = mock_graph
        
        # Mock input tensors
        mock_input = mock.MagicMock()
        mock_input.name = "input"
        mock_input.type.tensor_type.shape.dim = [
            mock.MagicMock(dim_param="batch_size", dim_value=0),
            mock.MagicMock(dim_param="", dim_value=10)
        ]
        mock_graph.input = [mock_input]
        
        # Mock output tensors
        mock_output = mock.MagicMock()
        mock_output.name = "output"
        mock_output.type.tensor_type.shape.dim = [
            mock.MagicMock(dim_param="batch_size", dim_value=0),
            mock.MagicMock(dim_param="", dim_value=5)
        ]
        mock_graph.output = [mock_output]
        
        # Mock nodes
        mock_node1 = mock.MagicMock()
        mock_node1.op_type = "Conv"
        mock_node2 = mock.MagicMock()
        mock_node2.op_type = "MatMul"
        mock_graph.node = [mock_node1, mock_node2]
        
        # Set up mocks
        mock_load.return_value = mock_model
        
        # Test checking model structure
        model_path = self.test_dir / 'model.onnx'
        metadata = check_model_structure(model_path)
        
        # Verify the result
        self.assertIn("inputs", metadata)
        self.assertIn("outputs", metadata)
        self.assertIn("op_types", metadata)
        
        # Check inputs
        self.assertEqual(len(metadata["inputs"]), 1)
        self.assertEqual(metadata["inputs"][0]["name"], "input")
        self.assertEqual(metadata["inputs"][0]["shape"], ["batch_size", 10])
        
        # Check outputs
        self.assertEqual(len(metadata["outputs"]), 1)
        self.assertEqual(metadata["outputs"][0]["name"], "output")
        self.assertEqual(metadata["outputs"][0]["shape"], ["batch_size", 5])
        
        # Check operators
        self.assertEqual(metadata["op_types"], {"Conv", "MatMul"})
        
        # Verify the mocks were called correctly
        mock_load.assert_called_once_with(str(model_path))
        mock_check.assert_called_once_with(mock_model)
    
    @mock.patch('onnxruntime.InferenceSession')
    def test_run_inference(self, mock_session_cls):
        """Test running inference on an ONNX model."""
        # Create mock objects
        mock_session = mock.MagicMock()
        mock_output1 = mock.MagicMock()
        mock_output1.name = "output1"
        mock_output2 = mock.MagicMock()
        mock_output2.name = "output2"
        
        # Set up mocks
        mock_session_cls.return_value = mock_session
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]
        mock_session.run.return_value = [
            np.ones((1, 5), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32)
        ]
        
        # Test running inference
        model_path = self.test_dir / 'model.onnx'
        dummy_inputs = {
            "input1": np.random.randn(1, 10).astype(np.float32),
            "input2": np.random.randn(1, 7).astype(np.float32)
        }
        
        outputs = run_inference(model_path, dummy_inputs)
        
        # Verify the result
        self.assertIn("output1", outputs)
        self.assertIn("output2", outputs)
        
        # Check output shapes
        self.assertEqual(outputs["output1"].shape, (1, 5))
        self.assertEqual(outputs["output2"].shape, (1, 3))
        
        # Verify the mocks were called correctly
        mock_session_cls.assert_called_once_with(str(model_path))
        mock_session.run.assert_called_once_with(["output1", "output2"], dummy_inputs)


if __name__ == '__main__':
    unittest.main()