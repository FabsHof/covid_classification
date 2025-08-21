"""
Environment utility for handling both Google Colab and local environments.
"""

import logging
from os import path
from typing import Tuple

from dotenv import dotenv_values

logger = logging.getLogger(__name__)


def mount_gdrive() -> bool:
    """
    Mounts Google Drive in Colab. Returns a boolean indicating success.

    Returns:
        bool: True if Google Drive was successfully mounted, False otherwise.
    """
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        logger.info("Google Drive mounted successfully.")
        return True
    except ImportError:
        logger.info("Not running in Google Colab.")
        return False


def setup_environment() -> Tuple[bool, dict, dict]:
    """
    Sets up the environment for both Google Colab and local environments.

    Returns:
        Tuple containing:
        - bool: True if running in Google Colab, False if local
        - dict: Configuration from .env file
        - dict: Environment-specific paths
    """
    # Detect environment and mount Google Drive if in Colab
    is_google_colab = mount_gdrive()

    # Load configuration from .env file
    env_file = (
        None if not is_google_colab else "drive/MyDrive/bds_covid_19/.env"
    )
    config = dotenv_values(env_file)

    # Setup paths based on environment
    if is_google_colab:
        data_path = config.get(
            "GDRIVE_DATA_PATH", "drive/MyDrive/bds_covid_19/data"
        )
        split_data_path = config.get(
            "GDRIVE_SPLIT_PATH", "drive/MyDrive/bds_covid_19/data_split"
        )
        models_path = config.get(
            "GDRIVE_MODELS_PATH", "drive/MyDrive/bds_covid_19/models"
        )
        image_path = config.get(
            "GDRIVE_IMAGE_PATH", "drive/MyDrive/bds_covid_19/data_split/images"
        )
    else:
        # For local environment, get root directory relative to this file
        root_dir = path.abspath(path.join(path.dirname(__file__), "..", ".."))
        data_path = path.join(root_dir, config.get("LOCAL_DATA_PATH", "data"))
        split_data_path = path.join(
            root_dir, config.get("LOCAL_SPLIT_PATH", "data_split")
        )
        models_path = path.join(
            root_dir, config.get("LOCAL_MODELS_PATH", "models")
        )
        image_path = path.join(
            root_dir, config.get("LOCAL_IMAGE_PATH", "data_split/images")
        )

    paths = {
        "data_path": data_path,
        "split_data_path": split_data_path,
        "models_path": models_path,
        "image_path": image_path,
        "test_path": (
            path.join(split_data_path, "images", "test")
            if split_data_path
            else None
        ),
    }

    logger.info(
        "Environment setup complete. Running in"
        f" {'Google Colab' if is_google_colab else 'local environment'}"
    )
    logger.info(f"Paths: {paths}")

    return is_google_colab, config, paths


def get_dataset_config() -> dict:
    """
    Returns the standard dataset configuration used across the project.

    Returns:
        dict: Dataset configuration with classes, categories, and split ratios
    """
    return {
        "classes": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
        "categories": ["images", "masks"],
        "split_ratio": (0.72, 0.18, 0.1),  # train, val, test
        "random_seed": 42,
    }
