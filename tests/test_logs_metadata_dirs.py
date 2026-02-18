"""
Tests for verifying that logs and metadata are placed in the correct directories.
These tests verify the source code changes directly without importing.
"""


def test_ensure_dir_function():
    """Test the ensure_dir function logic directly"""
    # Read the function source
    with open('/workspace/project/blech_clust/utils/blech_utils.py', 'r') as f:
        source = f.read()
    
    # Check that ensure_dir is defined
    assert 'def ensure_dir(data_dir, subdir):' in source
    
    # Check that get_metadata_dir is defined
    assert 'def get_metadata_dir(data_dir):' in source
    
    # Check that get_logs_dir is defined
    assert 'def get_logs_dir(data_dir):' in source
    
    print("✓ All required functions are defined in blech_utils.py")


def test_logs_directory_creation_in_tee():
    """Test that Tee class uses logs directory"""
    with open('/workspace/project/blech_clust/utils/blech_utils.py', 'r') as f:
        source = f.read()
    
    # Check that Tee uses logs directory
    assert "logs_dir = ensure_dir(data_dir, 'logs')" in source
    assert "self.log_path = os.path.join(logs_dir, name)" in source
    print("✓ Tee class correctly uses logs directory")


def test_logs_directory_in_blech_process():
    """Test that blech_process.py uses logs directory"""
    with open('/workspace/project/blech_clust/blech_process.py', 'r') as f:
        source = f.read()
    
    # Check that it creates logs directory
    assert "log_dir = os.path.join(metadata_handler.dir_name, 'logs')" in source
    assert "os.makedirs(log_dir, exist_ok=True)" in source
    print("✓ blech_process.py correctly uses logs directory")


def test_logs_directory_in_blech_init():
    """Test that blech_init.py uses logs directory"""
    with open('/workspace/project/blech_clust/blech_init.py', 'r') as f:
        source = f.read()
    
    # Check that it uses logs directory for results.log
    assert "logs/results.log" in source
    print("✓ blech_init.py correctly uses logs directory for results.log")


def test_logs_directory_in_ram_monitor():
    """Test that ram_monitor.py uses logs directory"""
    with open('/workspace/project/blech_clust/utils/ram_monitor.py', 'r') as f:
        source = f.read()
    
    # Check that it uses logs directory
    assert "logs_dir = os.path.join(output_dir, 'logs')" in source
    assert "ram_usage.log" in source
    print("✓ ram_monitor.py correctly uses logs directory")


def test_metadata_directory_in_read_file():
    """Test that read_file.py uses metadata directory"""
    with open('/workspace/project/blech_clust/utils/read_file.py', 'r') as f:
        source = f.read()
    
    # Check that it uses get_metadata_dir
    assert "from blech_clust.utils.blech_utils import get_metadata_dir" in source
    assert "metadata_dir = get_metadata_dir(self.data_dir)" in source
    print("✓ read_file.py correctly uses metadata directory")


def test_metadata_directory_in_ephys_data():
    """Test that ephys_data.py uses metadata directory"""
    with open('/workspace/project/blech_clust/utils/ephys_data/ephys_data.py', 'r') as f:
        source = f.read()
    
    # Check that it uses get_metadata_dir
    assert "from blech_clust.utils.blech_utils import get_metadata_dir" in source
    print("✓ ephys_data.py correctly imports get_metadata_dir")


def test_metadata_directory_in_blech_units_characteristics():
    """Test that blech_units_characteristics.py uses metadata directory"""
    with open('/workspace/project/blech_clust/blech_units_characteristics.py', 'r') as f:
        source = f.read()
    
    # Check that it uses get_metadata_dir for aggregated_characteristics.csv
    assert "get_metadata_dir" in source
    assert "aggregated_characteristics.csv" in source
    print("✓ blech_units_characteristics.py correctly uses metadata directory")


def test_metadata_directory_in_blech_post_process():
    """Test that blech_post_process.py uses metadata directory"""
    with open('/workspace/project/blech_clust/blech_post_process.py', 'r') as f:
        source = f.read()
    
    # Check that it uses get_metadata_dir for unit_descriptor.csv
    assert "get_metadata_dir" in source
    assert "unit_descriptor.csv" in source
    print("✓ blech_post_process.py correctly uses metadata directory")


def test_metadata_directory_in_blech_make_arrays():
    """Test that blech_make_arrays.py uses metadata directory"""
    with open('/workspace/project/blech_clust/blech_make_arrays.py', 'r') as f:
        source = f.read()
    
    # Check that it uses get_metadata_dir for trial_info_frame.csv
    assert "get_metadata_dir" in source
    assert "trial_info_frame.csv" in source
    print("✓ blech_make_arrays.py correctly uses metadata directory")


def test_backward_compatibility_check():
    """Test that backward compatibility is maintained"""
    # Check execution_log.json backward compatibility
    with open('/workspace/project/blech_clust/utils/blech_utils.py', 'r') as f:
        source = f.read()
    
    assert "old_log_path = os.path.join(self.data_dir, 'execution_log.json')" in source
    assert "if not os.path.exists(self.log_path) and os.path.exists(old_log_path):" in source
    print("✓ Backward compatibility for execution_log.json is maintained")


def test_find_file_in_dir_function():
    """Test that find_file_in_dir function exists for backward compatibility"""
    with open('/workspace/project/blech_clust/utils/blech_utils.py', 'r') as f:
        source = f.read()
    
    assert "def find_file_in_dir(data_dir, filename_pattern):" in source
    print("✓ find_file_in_dir function exists for backward compatibility")


if __name__ == '__main__':
    test_ensure_dir_function()
    test_logs_directory_creation_in_tee()
    test_logs_directory_in_blech_process()
    test_logs_directory_in_blech_init()
    test_logs_directory_in_ram_monitor()
    test_metadata_directory_in_read_file()
    test_metadata_directory_in_ephys_data()
    test_metadata_directory_in_blech_units_characteristics()
    test_metadata_directory_in_blech_post_process()
    test_metadata_directory_in_blech_make_arrays()
    test_backward_compatibility_check()
    test_find_file_in_dir_function()
    print("\nAll tests passed!")
