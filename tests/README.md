# Tests Directory

This directory contains unit tests and integration tests for the project.

## Structure

- `test_data_loader.py`: Tests for data loading utilities
- `test_training.py`: Tests for training pipeline
- `test_inference.py`: Tests for inference functionality
- `test_metrics.py`: Tests for metric calculations

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_loader.py

# Run with coverage
pytest --cov=src tests/
```

## Writing Tests

Follow pytest conventions:
- Test files should start with `test_`
- Test functions should start with `test_`
- Use fixtures for common setup/teardown
