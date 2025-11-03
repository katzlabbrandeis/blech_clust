"""
Create a dummy permanent dataset copy for testing permanent-path functionality.

This script creates a minimal copy of a test dataset with only the files
required to pass permanent-path validation checks in blech_exp_info.py.

The permanent-path validation checks for:
- info.rhd
- time.dat
- amplifier.dat

This script creates dummy versions of these files to satisfy the validation
without copying the entire dataset.
"""

import argparse
import os
import shutil


def create_dummy_file(filepath, size_bytes=1024):
    """
    Create a dummy file with specified size.

    Args:
        filepath: Path to the file to create
        size_bytes: Size of the file in bytes (default: 1KB)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(b'\x00' * size_bytes)
    print(f"Created dummy file: {filepath} ({size_bytes} bytes)")


def create_dummy_permanent_dataset(source_dir, permanent_base_dir):
    """
    Create a dummy permanent dataset with minimum required files.

    Creates a nested directory structure so the final directory name matches
    the source directory name (e.g., permanent_base/KM45_5tastes_210620_113227_new/)

    Args:
        source_dir: Source data directory (for reference)
        permanent_base_dir: Base directory for permanent storage
    """
    # Get the source directory name
    source_dir_name = os.path.basename(source_dir.rstrip('/'))

    # Create nested permanent directory with matching name
    permanent_dir = os.path.join(permanent_base_dir, source_dir_name)
    os.makedirs(permanent_dir, exist_ok=True)
    print(f"Created permanent directory: {permanent_dir}")

    # Required files for permanent-path validation
    required_files = ['info.rhd', 'time.dat', 'amplifier.dat']

    for filename in required_files:
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(permanent_dir, filename)

        # If source file exists, copy it (for info.rhd which is small)
        # Otherwise create a dummy file
        if os.path.exists(source_file) and filename == 'info.rhd':
            shutil.copy2(source_file, dest_file)
            print(f"Copied {filename} from source")
        else:
            # Create dummy files for large data files
            # Use small sizes for testing purposes
            if filename == 'info.rhd':
                size = 1024  # 1KB
            elif filename == 'time.dat':
                size = 1024 * 10  # 10KB
            elif filename == 'amplifier.dat':
                size = 1024 * 10  # 10KB
            else:
                size = 1024

            create_dummy_file(dest_file, size)

    print(f"\nDummy permanent dataset created at: {permanent_dir}")
    print("This directory contains minimal files to pass permanent-path validation.")

    return permanent_dir


def main():
    parser = argparse.ArgumentParser(
        description='Create dummy permanent dataset for testing'
    )
    parser.add_argument(
        'source_dir',
        help='Source data directory'
    )
    parser.add_argument(
        'permanent_base_dir',
        help='Base directory for permanent storage (will create nested dir with matching name)'
    )

    args = parser.parse_args()

    # Validate source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory does not exist: {args.source_dir}")
        return 1

    # Create dummy permanent dataset
    permanent_dir = create_dummy_permanent_dataset(
        args.source_dir, args.permanent_base_dir)

    # Print the final path for use in scripts
    print(f"PERMANENT_PATH={permanent_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
