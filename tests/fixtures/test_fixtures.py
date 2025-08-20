"""
Test fixtures and utilities for testing file system operations.
"""

import os
import shutil
import tempfile
from typing import Dict, Generator, List

import pytest


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataset_structure(temp_dir: str) -> Dict[str, str]:
    """
    Create a sample dataset structure for testing.

    Returns:
        Dict with paths to the created directories and files
    """
    # Create source dataset structure
    source_dir = os.path.join(temp_dir, "data")
    classes = ["COVID", "Normal", "Pneumonia"]
    categories = ["images", "masks"]

    structure = {
        "source_dir": source_dir,
        "target_dir": os.path.join(temp_dir, "data_split"),
        "classes": classes,
        "categories": categories,
        "files": {},
    }

    # Create directories and sample files
    for cls in classes:
        structure["files"][cls] = {}
        for category in categories:
            dir_path = os.path.join(source_dir, cls, category)
            os.makedirs(dir_path, exist_ok=True)

            # Create sample files (5 files per class/category)
            files = []
            for i in range(5):
                filename = f"{cls}-{i}.png"
                file_path = os.path.join(dir_path, filename)

                # Create simple test files without cv2 dependency
                with open(file_path, "wb") as f:
                    if category == "images":
                        f.write(b"FAKE_IMAGE_DATA_FOR_TESTING")
                    else:
                        f.write(b"FAKE_MASK_DATA_FOR_TESTING")

                files.append(filename)

            structure["files"][cls][category] = files

    return structure


@pytest.fixture
def sample_env_config(temp_dir: str) -> Dict[str, str]:
    """Create sample environment configuration."""
    return {
        "LOCAL_DATA_PATH": os.path.join(temp_dir, "data"),
        "LOCAL_SPLIT_PATH": os.path.join(temp_dir, "data_split"),
        "LOCAL_MODELS_PATH": os.path.join(temp_dir, "models"),
        "LOCAL_IMAGE_PATH": os.path.join(temp_dir, "data_split", "images"),
        "GDRIVE_DATA_PATH": "drive/MyDrive/test/data",
        "GDRIVE_SPLIT_PATH": "drive/MyDrive/test/data_split",
        "GDRIVE_MODELS_PATH": "drive/MyDrive/test/models",
        "GDRIVE_IMAGE_PATH": "drive/MyDrive/test/data_split/images",
    }


@pytest.fixture
def mock_env_file(temp_dir: str, sample_env_config: Dict[str, str]) -> str:
    """Create a mock .env file for testing."""
    env_file = os.path.join(temp_dir, ".env")
    with open(env_file, "w") as f:
        for key, value in sample_env_config.items():
            f.write(f"{key}={value}\n")
    return env_file


def assert_directory_structure_exists(
    base_path: str, expected_structure: Dict
) -> None:
    """Assert that a directory structure exists as expected."""
    assert os.path.exists(base_path), f"Base path {base_path} does not exist"

    for category in expected_structure.get("categories", []):
        category_path = os.path.join(base_path, category)
        assert os.path.exists(
            category_path
        ), f"Category path {category_path} does not exist"

        for split in ["train", "val", "test"]:
            split_path = os.path.join(category_path, split)
            assert os.path.exists(
                split_path
            ), f"Split path {split_path} does not exist"

            for cls in expected_structure.get("classes", []):
                class_path = os.path.join(split_path, cls)
                assert os.path.exists(
                    class_path
                ), f"Class path {class_path} does not exist"


def count_files_in_directory(
    directory: str, extensions: List[str] = None
) -> int:
    """Count files in a directory with optional extension filtering."""
    if not os.path.exists(directory):
        return 0

    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg"]

    count = 0
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            count += 1

    return count


def create_test_image(directory, filename, dimensions):
    """Create a test image file with specified dimensions."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    # Use simple file creation instead of cv2.imwrite for testing
    # Create a minimal image-like file for testing purposes
    with open(file_path, "wb") as f:
        f.write(b"FAKE_IMAGE_DATA_FOR_TESTING")

    return file_path


def create_test_mask(directory, filename, dimensions):
    """Create a test mask file with specified dimensions."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    # Create a simple binary mask-like file for testing
    with open(file_path, "wb") as f:
        f.write(b"FAKE_MASK_DATA_FOR_TESTING")

    return file_path
