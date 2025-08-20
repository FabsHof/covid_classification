# Testing Infrastructure

This directory contains comprehensive unit tests for the COVID classification project's utility modules.

## Overview

The test suite covers three main utility modules:
- `src/util/environment.py` - Environment detection and configuration
- `src/util/dataset.py` - Dataset restructuring and splitting
- `src/util/image_processing.py` - Image and mask processing

## Structure

````markdown
# 🧪 Testing Infrastructure

Comprehensive unit tests for the COVID classification project's utility modules.

## 🎯 Overview

Tests cover three main areas:
- 🌍 **Environment detection** - Google Colab vs local setup
- 📁 **Dataset processing** - Restructuring and splitting
- 🖼️ **Image operations** - Mask application and processing

## 🚀 Quick Start

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# With coverage
uv run pytest --cov=src --cov-report=html
```

## 📂 Structure

```
tests/
├── fixtures/          # 🔧 Shared test utilities
└── util/              # 🧰 Core module tests
    ├── test_environment.py
    ├── test_dataset.py
    └── test_image_processing.py
```

## ✨ Features

- 🎭 **Comprehensive mocking** - External dependencies (OpenCV, file I/O)
- 📁 **Temporary directories** - Clean filesystem testing
- 🔄 **Reproducible tests** - Consistent results with random seeds
- ⚡ **Fast execution** - Lightweight test data and mocking
- 🛡️ **Error handling** - Edge cases and failure scenarios

## 🤖 CI/CD Integration

Tests run automatically on main branch merges via pre-commit hooks to maintain code quality while keeping development fast.

## 💡 Best Practices

- ✅ Each test uses fresh temporary directories
- ✅ Mock external dependencies for isolation
- ✅ Test both success and failure paths
- ✅ Use minimal, realistic test data
````

## Test Fixtures

The `tests/fixtures/test_fixtures.py` module provides:

- **`temp_dir`**: Creates temporary directories for filesystem testing
- **`sample_dataset_structure`**: Sets up realistic dataset structures for testing
- **`mock_env_config`**: Provides mock environment configurations
- **Utility functions**: Helper functions for file creation and validation

## Running Tests

### Quick Test Run
```bash
python run_tests.py
```

### Using pytest directly
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/util/test_environment.py -v

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Using uv
```bash
# Install test dependencies and run
uv sync
uv run pytest tests/
```

## Test Categories

### Environment Tests (`test_environment.py`)
- Mount Google Drive functionality
- Environment setup and detection
- Dataset configuration loading
- Path resolution and validation
- Error handling for missing configurations

### Dataset Tests (`test_dataset.py`)
- Dataset restructuring with proper file correspondence
- Stratified splitting across train/validation/test
- File distribution verification
- Split consistency across image categories
- Directory cleaning and creation
- Reproducibility with random seeds

### Image Processing Tests (`test_image_processing.py`)
- Mask application to images
- Dimension mismatch handling
- Batch processing of masked images
- Directory structure creation
- Error handling for missing files
- Performance and logging verification

## Test Patterns

### Filesystem Testing
Tests use temporary directories and realistic file structures:

```python
def test_function(sample_dataset_structure):
    source_dir = sample_dataset_structure["source_dir"]
    target_dir = sample_dataset_structure["target_dir"]
    # Test filesystem operations...
```

### Mocking External Dependencies
Tests mock external libraries (OpenCV, etc.) for isolation:

```python
with patch('cv2.imread') as mock_imread:
    mock_imread.return_value = test_image
    # Test image processing...
```

### Error Condition Testing
Tests cover various error scenarios:

```python
def test_missing_file_handling(temp_dir):
    nonexistent_file = os.path.join(temp_dir, "missing.png")
    result = process_image(nonexistent_file)
    assert result is None
```

## Coverage Goals

The test suite aims for:
- **80%+ code coverage** across all utility modules
- **100% function coverage** for public interfaces
- **Comprehensive error handling** testing
- **Edge case validation** (empty directories, malformed files, etc.)

## Pre-commit Integration

Tests are automatically run via pre-commit hooks:

```yaml
- repo: local
  hooks:
    - id: pytest
      name: pytest
      entry: uv run pytest
      language: system
      types: [python]
      pass_filenames: false
      always_run: true
```

To install pre-commit hooks:
```bash
pre-commit install
```

## Best Practices

### Test Isolation
- Each test uses fresh temporary directories
- Mock external dependencies
- Clean up resources after test completion

### Test Data
- Use realistic but minimal test datasets
- Create test files programmatically
- Avoid committing large binary test files

### Error Testing
- Test both success and failure paths
- Validate error messages and types
- Ensure graceful handling of edge cases

### Performance
- Use fast, lightweight test data
- Mock slow operations (file I/O, network calls)
- Set reasonable timeouts for long-running tests

## Troubleshooting

### Import Errors
If you see import errors, ensure dependencies are installed:
```bash
uv sync
```

### File Permission Errors
Ensure the test runner has write permissions to the temporary directory.

### Mock Failures
If mocks aren't working as expected, verify the import paths match the actual module structure.

## Contributing

When adding new utility functions:

1. **Write tests first** (TDD approach)
2. **Cover happy path and edge cases**
3. **Use appropriate fixtures** from `test_fixtures.py`
4. **Add docstrings** explaining test purpose
5. **Mock external dependencies** appropriately
6. **Verify tests pass** before committing

## Configuration

Test configuration is managed in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "--cov=src",
    "--cov-report=html",
    "--cov-fail-under=80"
]
```
