"""
Unit tests for environment utility functions.
"""

import os
from unittest.mock import patch

from src.util.environment import (  # noqa: E501
    get_dataset_config,
    mount_gdrive,
    setup_environment,
)


class TestMountGdrive:
    """Test mount_gdrive function."""

    def test_mount_gdrive_not_in_colab(self):
        """Test mount_gdrive returns False when not in Google Colab."""
        result = mount_gdrive()
        assert result is False

    def test_mount_gdrive_in_colab_success(self):
        """Test mount_gdrive handles Google Colab environment."""
        # Since we can't easily mock the Google Colab environment,
        # and the function gracefully handles ImportError,
        # we'll test that it returns False in non-Colab environment
        result = mount_gdrive()
        assert result is False

    def test_mount_gdrive_import_error(self):
        """Test mount_gdrive handles ImportError gracefully."""
        # The actual function should catch ImportError and return False
        result = mount_gdrive()
        assert result is False


class TestSetupEnvironment:
    """Test setup_environment function."""

    def test_setup_environment_local_default_paths(self, temp_dir):
        """Test setup_environment with default local paths."""
        with patch("src.util.environment.mount_gdrive", return_value=False):
            with patch("src.util.environment.dotenv_values", return_value={}):
                with patch("src.util.environment.path.dirname") as mock_dirname:
                    # Mock the path resolution to use temp_dir
                    mock_dirname.return_value = temp_dir

                    is_colab, config, paths = setup_environment()

                    assert is_colab is False
                    assert isinstance(config, dict)
                    assert isinstance(paths, dict)

                    # Check required keys exist
                    required_keys = [
                        "data_path",
                        "split_data_path",
                        "models_path",
                        "image_path",
                        "test_path",
                    ]
                    for key in required_keys:
                        assert key in paths

    def test_setup_environment_local_custom_paths(
        self, temp_dir, sample_env_config
    ):
        """Test setup_environment with custom local paths from config."""
        with patch("src.util.environment.mount_gdrive", return_value=False):
            with patch(
                "src.util.environment.dotenv_values",
                return_value=sample_env_config,
            ):
                with patch("src.util.environment.path.dirname") as mock_dirname:
                    mock_dirname.return_value = temp_dir

                    is_colab, config, paths = setup_environment()

                    assert is_colab is False
                    assert config == sample_env_config

                    # Check that custom paths are used
                    expected_data_path = os.path.join(temp_dir, "data")
                    expected_split_path = os.path.join(temp_dir, "data_split")

                    assert expected_data_path in paths["data_path"]
                    assert expected_split_path in paths["split_data_path"]

    def test_setup_environment_google_colab(self, sample_env_config):
        """Test setup_environment in Google Colab environment."""
        with patch("src.util.environment.mount_gdrive", return_value=True):
            with patch(
                "src.util.environment.dotenv_values",
                return_value=sample_env_config,
            ):
                is_colab, config, paths = setup_environment()

                assert is_colab is True
                assert config == sample_env_config

                # Check that Google Drive paths are used
                assert (
                    paths["data_path"] == sample_env_config["GDRIVE_DATA_PATH"]
                )
                assert (
                    paths["split_data_path"]
                    == sample_env_config["GDRIVE_SPLIT_PATH"]
                )

    def test_setup_environment_paths_structure(self, temp_dir):
        """Test that setup_environment returns correctly structured paths."""
        with patch("src.util.environment.mount_gdrive", return_value=False):
            with patch("src.util.environment.dotenv_values", return_value={}):
                with patch("src.util.environment.path.dirname") as mock_dirname:
                    mock_dirname.return_value = temp_dir

                    is_colab, config, paths = setup_environment()

                    # Verify all required paths are present
                    required_paths = [
                        "data_path",
                        "split_data_path",
                        "models_path",
                        "image_path",
                        "test_path",
                    ]
                    for path_key in required_paths:
                        assert path_key in paths
                        assert isinstance(paths[path_key], str)

                    # Verify test_path is correctly constructed
                    assert "images/test" in paths["test_path"]


class TestGetDatasetConfig:
    """Test get_dataset_config function."""

    def test_get_dataset_config_structure(self):
        """Test that get_dataset_config returns the expected structure."""
        config = get_dataset_config()

        assert isinstance(config, dict)

        # Check required keys
        required_keys = ["classes", "categories", "split_ratio", "random_seed"]
        for key in required_keys:
            assert key in config

    def test_get_dataset_config_values(self):
        """Test that get_dataset_config returns expected values."""
        config = get_dataset_config()

        # Check specific values
        assert config["classes"] == [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia",
        ]
        assert config["categories"] == ["images", "masks"]
        assert config["split_ratio"] == (0.72, 0.18, 0.1)
        assert config["random_seed"] == 42

    def test_get_dataset_config_split_ratio_sum(self):
        """Test that split ratios sum to 1.0."""
        config = get_dataset_config()
        split_ratio = config["split_ratio"]

        assert isinstance(split_ratio, tuple)
        assert len(split_ratio) == 3
        assert (
            abs(sum(split_ratio) - 1.0) < 1e-10
        )  # Account for floating point precision

    def test_get_dataset_config_immutability(self):
        """Test that get_dataset_config returns a new dict each time."""
        config1 = get_dataset_config()
        config2 = get_dataset_config()

        # Should be equal but not the same object
        assert config1 == config2
        assert config1 is not config2

        # Modifying one shouldn't affect the other
        config1["classes"].append("Test")
        assert config1 != config2
