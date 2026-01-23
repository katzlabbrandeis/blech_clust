"""
Tests for optional dependency functionality in the pipeline graph check system.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, mock_open
import sys

# Add the parent directory to sys.path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.blech_utils import pipeline_graph_check


class TestOptionalDependencies:
    """Test cases for optional dependency functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data_dir = tempfile.mkdtemp()
        self.test_blech_dir = tempfile.mkdtemp()
        
        # Create a sample dependency graph with optional dependencies
        self.sample_graph = {
            "graph": {
                "": {
                    "blech_init.py": "blech_exp_info.py",
                    "blech_common_avg_reference.py": "blech_init.py",
                    "blech_make_arrays.py": "blech_init.py",
                    "blech_process.py": {
                        "optional": True,
                        "scripts": ["blech_common_avg_reference.py"]
                    },
                    "blech_post_process.py": "blech_process.py"
                }
            }
        }
        
        # Create sample script files
        for script in ["blech_init.py", "blech_exp_info.py", "blech_common_avg_reference.py", 
                      "blech_make_arrays.py", "blech_process.py", "blech_post_process.py"]:
            with open(os.path.join(self.test_blech_dir, script), 'w') as f:
                f.write("# Sample script file\n")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
        shutil.rmtree(self.test_blech_dir, ignore_errors=True)

    @patch('utils.blech_utils.path_handler')
    def test_graph_loading_with_optional_dependencies(self, mock_path_handler):
        """Test that the dependency graph loads correctly with optional dependencies."""
        # Mock the path handler to return our test directory
        mock_handler_instance = mock_path_handler.return_value
        mock_handler_instance.blech_clust_dir = self.test_blech_dir
        
        # Create the dependency graph file
        graph_path = os.path.join(self.test_blech_dir, 'params', 'dependency_graph.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(self.sample_graph, f)
        
        # Create the pipeline check
        checker = pipeline_graph_check(self.test_data_dir)
        
        # Verify the graph was loaded
        assert '' in checker.graph
        assert 'blech_process.py' in checker.graph['']

    @patch('utils.blech_utils.path_handler')
    def test_optional_dependency_detection(self, mock_path_handler):
        """Test that optional dependencies are correctly identified."""
        # Mock the path handler
        mock_handler_instance = mock_path_handler.return_value
        mock_handler_instance.blech_clust_dir = self.test_blech_dir
        
        # Create the dependency graph file
        graph_path = os.path.join(self.test_blech_dir, 'params', 'dependency_graph.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(self.sample_graph, f)
        
        checker = pipeline_graph_check(self.test_data_dir)
        
        # Test optional dependency detection
        parent_script = os.path.join(self.test_blech_dir, 'blech_common_avg_reference.py')
        child_script = os.path.join(self.test_blech_dir, 'blech_process.py')
        
        # This should be optional
        assert checker._is_optional_dependency(parent_script, child_script) == True
        
        # Test non-optional dependency
        parent_script2 = os.path.join(self.test_blech_dir, 'blech_process.py')
        child_script2 = os.path.join(self.test_blech_dir, 'blech_post_process.py')
        
        # This should not be optional
        assert checker._is_optional_dependency(parent_script2, child_script2) == False

    @patch('utils.blech_utils.path_handler')
    def test_check_previous_with_optional_dependency_missing(self, mock_path_handler):
        """Test check_previous when optional dependency is missing."""
        # Mock the path handler
        mock_handler_instance = mock_path_handler.return_value
        mock_handler_instance.blech_clust_dir = self.test_blech_dir
        
        # Create the dependency graph file
        graph_path = os.path.join(self.test_blech_dir, 'params', 'dependency_graph.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(self.sample_graph, f)
        
        # Create execution log with no optional dependency completed
        execution_log = {
            "completed": {
                os.path.join(self.test_blech_dir, 'blech_init.py'): "2024-01-01 10:00:00",
                os.path.join(self.test_blech_dir, 'blech_make_arrays.py'): "2024-01-01 10:05:00"
            }
        }
        
        log_path = os.path.join(self.test_data_dir, 'execution_log.json')
        with open(log_path, 'w') as f:
            json.dump(execution_log, f)
        
        checker = pipeline_graph_check(self.test_data_dir)
        
        # Test blech_process.py - should succeed even without optional dependency
        script_path = os.path.join(self.test_blech_dir, 'blech_process.py')
        
        # This should not raise an exception
        result = checker.check_previous(script_path)
        assert result == True

    @patch('utils.blech_utils.path_handler')
    def test_check_previous_with_optional_dependency_completed(self, mock_path_handler):
        """Test check_previous when optional dependency is completed."""
        # Mock the path handler
        mock_handler_instance = mock_path_handler.return_value
        mock_handler_instance.blech_clust_dir = self.test_blech_dir
        
        # Create the dependency graph file
        graph_path = os.path.join(self.test_blech_dir, 'params', 'dependency_graph.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(self.sample_graph, f)
        
        # Create execution log with optional dependency completed
        execution_log = {
            "completed": {
                os.path.join(self.test_blech_dir, 'blech_init.py'): "2024-01-01 10:00:00",
                os.path.join(self.test_blech_dir, 'blech_common_avg_reference.py'): "2024-01-01 10:02:00",
                os.path.join(self.test_blech_dir, 'blech_make_arrays.py'): "2024-01-01 10:05:00"
            }
        }
        
        log_path = os.path.join(self.test_data_dir, 'execution_log.json')
        with open(log_path, 'w') as f:
            json.dump(execution_log, f)
        
        checker = pipeline_graph_check(self.test_data_dir)
        
        # Test blech_process.py - should succeed with optional dependency completed
        script_path = os.path.join(self.test_blech_dir, 'blech_process.py')
        
        # This should not raise an exception
        result = checker.check_previous(script_path)
        assert result == True

    @patch('utils.blech_utils.path_handler')
    def test_check_previous_with_required_dependency_missing(self, mock_path_handler):
        """Test check_previous when required dependency is missing (should fail)."""
        # Mock the path handler
        mock_handler_instance = mock_path_handler.return_value
        mock_handler_instance.blech_clust_dir = self.test_blech_dir
        
        # Create the dependency graph file
        graph_path = os.path.join(self.test_blech_dir, 'params', 'dependency_graph.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(self.sample_graph, f)
        
        # Create execution log missing required dependency
        execution_log = {
            "completed": {
                os.path.join(self.test_blech_dir, 'blech_init.py'): "2024-01-01 10:00:00"
            }
        }
        
        log_path = os.path.join(self.test_data_dir, 'execution_log.json')
        with open(log_path, 'w') as f:
            json.dump(execution_log, f)
        
        checker = pipeline_graph_check(self.test_data_dir)
        
        # Test blech_post_process.py - should fail without required dependency
        script_path = os.path.join(self.test_blech_dir, 'blech_post_process.py')
        
        # This should raise a ValueError because blech_process.py is required but not completed
        with pytest.raises(ValueError, match="Required parent script"):
            checker.check_previous(script_path)

    @patch('utils.blech_utils.path_handler')
    def test_graph_flattening_with_optional_dependencies(self, mock_path_handler):
        """Test that the graph is flattened correctly with optional dependencies."""
        # Mock the path handler
        mock_handler_instance = mock_path_handler.return_value
        mock_handler_instance.blech_clust_dir = self.test_blech_dir
        
        # Create the dependency graph file
        graph_path = os.path.join(self.test_blech_dir, 'params', 'dependency_graph.json')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        with open(graph_path, 'w') as f:
            json.dump(self.sample_graph, f)
        
        checker = pipeline_graph_check(self.test_data_dir)
        
        # Verify flat_graph structure
        expected_parent = os.path.join(self.test_blech_dir, 'blech_process.py')
        expected_child = os.path.join(self.test_blech_dir, 'blech_common_avg_reference.py')
        
        assert expected_parent in checker.flat_graph
        assert expected_child in checker.flat_graph[expected_parent]
        assert isinstance(checker.flat_graph[expected_parent], list)

if __name__ == "__main__":
    pytest.main([__file__])
