import logging
from os import path

import albumentations
from dotenv import dotenv_values

from src.util import dataset

# Setup logging
logging.basicConfig(level=logging.INFO)


# ################
# Setup
# (intended to run locally)
# ################
def main():

    # Environment variables
    config = dotenv_values()
    root_dir = path.abspath(path.join(path.dirname(__file__), "..", ".."))
    source_dir = path.join(root_dir, config.get("LOCAL_DATA_PATH", "data"))
    target_dir = path.join(
        root_dir, config.get("LOCAL_SPLIT_PATH", "data_split")
    )

    classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    categories = ["images", "masks"]
    split_ratio = (0.72, 0.18, 0.1)  # train, val, test
    random_seed = 42

    # Restructure dataset
    dataset.restructure_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        dataset_classes=classes,
        dataset_categories=categories,
        split_ratio=split_ratio,
        random_seed=random_seed,
    )

    # Augment dataset
    dataset.augment_dataset(
        source_dir=target_dir,
        augmentations=[
            albumentations.RandomBrightnessContrast(),
            albumentations.HorizontalFlip(),
            albumentations.Rotate(limit=15),
            albumentations.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=10
            ),
            albumentations.OneOf(
                [
                    albumentations.GaussianBlur(blur_limit=3),
                    albumentations.GaussNoise(var_limit=(10.0, 50.0)),
                ],
                p=0.3,
            ),
        ],
        dataset_classes=classes,
        dataset_categories=categories,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    main()
