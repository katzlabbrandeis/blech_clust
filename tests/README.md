# Tests for blech_clust

This directory contains pytest tests for the blech_clust.py script and related functionality.

## Test Structure

- `conftest.py`: Contains pytest fixtures used across multiple test files
- `test_blech_clust.py`: Tests for the main classes and functions in blech_clust.py
- `test_main.py`: Tests for the main function flow
- `test_process_functions.py`: Tests for data processing functions

## Running Tests

To run all tests:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_blech_clust.py
```

To run with verbose output:

```bash
pytest -v tests/
```

## Test Coverage

These tests cover:

1. Argument parsing
2. HDF5 file handling
3. Directory creation
4. File list processing
5. Data processing functions
6. Parameter file creation
7. Quality assurance functions
8. Main function flow

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of test classes and functions
2. Use fixtures from conftest.py where appropriate
3. Use mocking to avoid actual file I/O or external dependencies
4. Add appropriate assertions to verify functionality
