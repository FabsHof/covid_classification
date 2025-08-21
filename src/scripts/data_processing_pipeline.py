import logging

from src.util.dataset import restructure_dataset
from src.util.environment import get_dataset_config, setup_environment
from src.util.image_processing import create_masked_images

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function for data transformation pipeline.
    Supports both local and Google Colab environments.
    """
    # Setup environment and get paths
    is_google_colab, config, paths = setup_environment()
    dataset_config = get_dataset_config()

    print("🚀 Starting data transformation pipeline...")
    print(f"📍 Environment: {'Google Colab' if is_google_colab else 'Local'}")
    print(f"📂 Source: {paths['data_path']}")
    print(f"📂 Target: {paths['split_data_path']}")

    # Restructure dataset
    print("🔄 Restructuring dataset...")
    restructure_dataset(
        source_dir=paths["data_path"],
        target_dir=paths["split_data_path"],
        dataset_classes=dataset_config["classes"],
        dataset_categories=dataset_config["categories"],
        split_ratio=dataset_config["split_ratio"],
        random_seed=dataset_config["random_seed"],
    )

    # Create masked images
    print("🎭 Creating masked images...")
    create_masked_images(
        split_data_path=paths["split_data_path"],
        dataset_classes=dataset_config["classes"],
    )

    print("✅ Data transformation pipeline completed successfully!")


if __name__ == "__main__":
    main()
