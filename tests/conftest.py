"""
Pytest configuration and shared fixtures.
"""

# Import all fixtures from the fixtures module to make them available globally
from tests.fixtures.test_fixtures import (
    assert_directory_structure_exists,
    count_files_in_directory,
    create_test_image,
    create_test_mask,
    mock_env_file,
    sample_dataset_structure,
    sample_env_config,
    temp_dir,
)
