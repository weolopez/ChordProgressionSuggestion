#!/usr/bin/env python
"""
Test the model download functionality.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from download_model import download_file, extract_tarfile, verify_saved_model


class MockResponse:
    """Mock requests.Response for testing."""
    
    def __init__(self, content, headers=None):
        self.content = content
        self.headers = headers or {}
        self._content_consumed = False
    
    def iter_content(self, chunk_size):
        """Yield content in chunks."""
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]
    
    def raise_for_status(self):
        """Mock successful response."""
        pass


class TestDownload(unittest.TestCase):
    """Test the model download functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path('./test_download')
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
    
    @mock.patch('requests.get')
    def test_download_file(self, mock_get):
        """Test downloading a file."""
        # Prepare mock response
        mock_content = b'test content'
        mock_get.return_value = MockResponse(
            mock_content,
            headers={'content-length': str(len(mock_content))}
        )
        
        # Test downloading a file
        output_path = self.test_dir / 'test_file.txt'
        download_file('https://example.com/test', output_path)
        
        # Check if the file was created with the right content
        self.assertTrue(output_path.exists())
        with open(output_path, 'rb') as f:
            self.assertEqual(f.read(), mock_content)
    
    def test_verify_saved_model(self):
        """Test verifying a SavedModel directory."""
        # Create a mock SavedModel directory
        model_dir = self.test_dir / 'saved_model'
        model_dir.mkdir(exist_ok=True)
        
        # Create mock files
        (model_dir / 'saved_model.pb').touch()
        variables_dir = model_dir / 'variables'
        variables_dir.mkdir(exist_ok=True)
        
        # Test verification
        self.assertTrue(verify_saved_model(model_dir))
        
        # Test verification failure (missing saved_model.pb)
        (model_dir / 'saved_model.pb').unlink()
        self.assertFalse(verify_saved_model(model_dir))
        
        # Test verification failure (missing variables directory)
        (model_dir / 'saved_model.pb').touch()
        variables_dir.rmdir()
        self.assertFalse(verify_saved_model(model_dir))


if __name__ == '__main__':
    unittest.main()