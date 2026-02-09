"""
Load test configuration from test_config.json
"""
import json
import os

_script_dir = os.path.dirname(os.path.realpath(__file__))
_config_path = os.path.join(_script_dir, 'test_config.json')


def load_test_config():
    """Load test configuration from test_config.json"""
    with open(_config_path, 'r') as f:
        return json.load(f)


def get_test_data_dir():
    """Get the base directory for test data"""
    config = load_test_config()
    return os.path.expanduser(config['test_data_dir'])


def get_dataset_info(file_type):
    """Get dataset info for a specific file type (ofpc or trad)"""
    config = load_test_config()
    return config['datasets'].get(file_type)


def get_data_dirs_dict():
    """Get dictionary mapping file types to full data directory paths"""
    config = load_test_config()
    base_dir = os.path.expanduser(config['test_data_dir'])
    return {
        key: os.path.join(base_dir, info['name'])
        for key, info in config['datasets'].items()
    }


def get_gdrive_ids():
    """Get list of Google Drive IDs for all datasets"""
    config = load_test_config()
    return [info['gdrive_id'] for info in config['datasets'].values()]


def get_dataset_names():
    """Get list of dataset directory names"""
    config = load_test_config()
    return [info['name'] for info in config['datasets'].values()]
