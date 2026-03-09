"""
Test for blech_reload_amp_digs.py to ensure dig-in loading has been removed.

This test verifies that the fix for issue #799 is correct - that 
blech_reload_amp_digs.py no longer attempts to load dig-ins since they
are now loaded into dig_in_frame via blech_init.py.
"""
import ast
import os
import pytest


class TestBlechReloadAmpDigs:
    """Test class for blech_reload_amp_digs module"""
    
    def test_no_read_digins_calls(self):
        """Test that read_digins is not called in blech_reload_amp_digs.py"""
        # Read the source file
        file_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'utils', 
            'blech_reload_amp_digs.py'
        )
        
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the source code
        tree = ast.parse(source)
        
        # Find all function calls
        read_digins_calls = []
        read_digins_single_file_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Check for read_file.read_digins
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'read_file' and
                        node.func.attr == 'read_digins'):
                        read_digins_calls.append(node.lineno)
                    # Check for read_file.read_digins_single_file
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'read_file' and
                        node.func.attr == 'read_digins_single_file'):
                        read_digins_single_file_calls.append(node.lineno)
        
        # Assert that there are no calls to read_digins or read_digins_single_file
        assert len(read_digins_calls) == 0, (
            f"Found {len(read_digins_calls)} call(s) to read_digins at line(s): "
            f"{read_digins_calls}. These should be removed as dig-ins are now "
            f"loaded via dig_in_frame in blech_init.py"
        )
        
        assert len(read_digins_single_file_calls) == 0, (
            f"Found {len(read_digins_single_file_calls)} call(s) to "
            f"read_digins_single_file at line(s): {read_digins_single_file_calls}. "
            f"These should be removed as dig-ins are now loaded via "
            f"dig_in_frame in blech_init.py"
        )

    def test_read_electrode_channels_still_present(self):
        """Test that read_electrode_channels is still called in the file"""
        # Read the source file
        file_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'utils', 
            'blech_reload_amp_digs.py'
        )
        
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the source code
        tree = ast.parse(source)
        
        # Find function calls to read_electrode_channels
        read_electrode_channels_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'read_file' and
                        node.func.attr == 'read_electrode_channels'):
                        read_electrode_channels_calls.append(node.lineno)
        
        # Assert that read_electrode_channels is still called
        assert len(read_electrode_channels_calls) > 0, (
            "read_electrode_channels should still be called in "
            "blech_reload_amp_digs.py"
        )

    def test_read_emg_channels_still_present(self):
        """Test that read_emg_channels is still called in the file"""
        # Read the source file
        file_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'utils', 
            'blech_reload_amp_digs.py'
        )
        
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the source code
        tree = ast.parse(source)
        
        # Find function calls to read_emg_channels
        read_emg_channels_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'read_file' and
                        node.func.attr == 'read_emg_channels'):
                        read_emg_channels_calls.append(node.lineno)
        
        # Assert that read_emg_channels is still called
        assert len(read_emg_channels_calls) > 0, (
            "read_emg_channels should still be called in "
            "blech_reload_amp_digs.py"
        )
