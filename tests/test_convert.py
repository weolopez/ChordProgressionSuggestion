#!/usr/bin/env python
"""
Test the model conversion functionality.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import onnx
from convert_model import check_for_custom_layers, optimize_onnx_model


class TestConvert(unittest.TestCase):
    """Test the model conversion functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path('./test_convert')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create an ONNX model for testing
        self.test_onnx_path = self.test_dir / 'test_model.onnx'
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            # Remove all files in the test directory
            for file in self.test_dir.glob('*'):
                if file.is_file():
                    file.unlink()
            
            # Remove the test directory
            self.test_dir.rmdir()
    
    @mock.patch('tensorflow.saved_model.load')
    def test_check_for_custom_layers(self, mock_load):
        """Test checking for custom layers."""
        # Create a mock model with a signature containing nodes
        mock_model = mock.MagicMock()
        mock_signature = mock.MagicMock()
        
        # Create mock nodes with different ops
        node1 = mock.MagicMock()
        node1.op = 'Conv2D'
        node2 = mock.MagicMock()
        node2.op = 'CustomAttention'
        
        # Set up the mock graph with the nodes
        mock_graph = mock.MagicMock()
        mock_graph.as_graph_def().node = [node1, node2]
        
        # Connect everything
        mock_signature.graph = mock_graph
        mock_model.signatures.values.return_value = [mock_signature]
        mock_load.return_value = mock_model
        
        # Test checking for custom layers
        custom_layers = check_for_custom_layers(Path('./dummy_path'))
        
        # Verify that the custom layer was detected
        self.assertIn('CustomAttention', custom_layers)
        self.assertNotIn('Conv2D', custom_layers)
    
    @mock.patch('onnx.load')
    @mock.patch('onnxsim.simplify')
    @mock.patch('onnx.save')
    def test_optimize_onnx_model(self, mock_save, mock_simplify, mock_load):
        """Test optimizing an ONNX model."""
        # Create mock objects
        mock_model = mock.MagicMock()
        mock_simplified_model = mock.MagicMock()
        
        # Set up mocks
        mock_load.return_value = mock_model
        mock_simplify.return_value = (mock_simplified_model, True)
        
        # Test optimizing an ONNX model
        input_path = self.test_dir / 'input.onnx'
        output_path = self.test_dir / 'output.onnx'
        
        result = optimize_onnx_model(input_path, output_path)
        
        # Verify the result
        self.assertEqual(result, output_path)
        
        # Verify the mocks were called correctly
        mock_load.assert_called_once_with(str(input_path))
        mock_simplify.assert_called_once_with(mock_model)
        mock_save.assert_called_once_with(mock_simplified_model, str(output_path))


if __name__ == '__main__':
    unittest.main()