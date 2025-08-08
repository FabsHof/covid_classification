import logging
import os
import random
import shutil
from os import path

import albumentations
import cv2

logger = logging.getLogger(__name__)


def restructure_dataset(
    source_dir,
    target_dir,
    dataset_classes=None,
    dataset_categories=None,
    split_ratio=(0.72, 0.18, 0.1),  # x% train, x% val, x% test
    random_seed=42,
):
    """
    Restructures the original Kaggle dataset into the structure that is used
    within this project. This includes copying images into the `target_dir`,
    reshaping the directory structure and splitting the dataset into train,
    validation and test sets (stratified). Therefore, the class ratios are
    preserved. Assumes the original dataset being of the following form:
    ```
    - source_dir
        - COVID
            - images
                - image1.jpg
                - image2.jpg
            - masks
                - image1_mask.jpg
                - image2_mask.jpg
        ...
    ```
    and restructures it to:
    ```
    - target_dir
        - images
            - test
                - COVID
                    - image1.jpg
                    - image2.jpg
            - train
            ...
            - val
            ...
    ```

    Args:
        source_dir (str): The path to the source directory containing the
            dataset.
        target_dir (str): The path to the target directory where the
            restructured dataset will be saved.
        dataset_classes (list): Optional list of classes, e.g. ['COVID',
            'Lung_Opacity']. If None, all classes found in the source
            directory will be used.
        dataset_categories (list): Optional list of categories, e.g. [
            'images', 'masks']. If None, all categories found in the
            source directory will be used.
        split_ratio (tuple): A tuple of three floats representing the
            ratio for train, validation, and test splits.
        random_seed (int): Random seed for reproducibility.
    Returns:
        None
    """
    logger.info(
        f"> Restructuring dataset from {source_dir} to {target_dir} with split"
        f" ratio {split_ratio}."
    )

    if not path.exists(source_dir):
        logger.error(f"\t- > The source directory {source_dir} does not exist.")
        raise ValueError(f"> The source directory {source_dir} does not exist.")
    if not path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Get dataset classes and categories if not provided
    if dataset_classes is None:
        dataset_classes = [
            d
            for d in os.listdir(source_dir)
            if path.isdir(path.join(source_dir, d))
        ]
    if dataset_categories is None:
        dataset_categories = [
            d
            for d in os.listdir(path.join(source_dir, dataset_classes[0]))
            if path.isdir(path.join(source_dir, dataset_classes[0], d))
        ]

    # Create target directories
    for category in dataset_categories:
        category_path = path.join(target_dir, category)
        if not path.exists(category_path):
            os.makedirs(category_path, exist_ok=True)
        for split in ["train", "val", "test"]:
            split_path = path.join(category_path, split)
            if not path.exists(split_path):
                os.makedirs(split_path, exist_ok=True)

    # Copy and split images
    for cls in dataset_classes:
        cls_path = path.join(source_dir, cls)
        if not path.exists(cls_path):
            logger.warning(
                f"\t- Class {cls} does not exist in source directory. Skipping."
            )
            continue

        for category in dataset_categories:
            category_path = path.join(cls_path, category)
            if not path.exists(category_path):
                logger.warning(
                    f"\t- Category {category} does not exist in class {cls}."
                    " Skipping."
                )
                continue

            # Get all image files in the category
            image_files = [
                f
                for f in os.listdir(category_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            random.shuffle(image_files)

            # Stratified split to keep class ratios
            n_total = len(image_files)
            n_train = int(n_total * split_ratio[0])
            n_val = int(n_total * split_ratio[1])
            n_test = n_total - n_train - n_val  # Ensure all files are used

            # Shuffle already done above
            train_files = image_files[:n_train]
            val_files = image_files[n_train : n_train + n_val]
            test_files = image_files[n_train + n_val : n_train + n_val + n_test]

            # Copy files to the respective directories
            for split, files in zip(
                ["train", "val", "test"], [train_files, val_files, test_files]
            ):
                split_dir = path.join(target_dir, category, split, cls)
                for file in files:
                    src_file = path.join(category_path, file)
                    dst_file = path.join(split_dir, file)
                    os.makedirs(path.dirname(dst_file), exist_ok=True)
                    shutil.copy(src_file, dst_file)
    logger.info(
        f"\t- Dataset restructuring completed. Files copied to {target_dir}."
    )


def augment_dataset(  # noqa: C901
    source_dir,
    augmentations,
    dataset_classes=None,
    dataset_categories=None,
    random_seed=42,
):
    """
    Augments the dataset in place by taking random original images and applying
    various transformations. Therefore, the dataset is analyzed for each
    category (e.g. 'images') and split ('train', 'val', 'test'). For each of
    these subsets, the maximum number of each class is determined. The
    augmentation is then applied to the images of each class in each category
    until the maximum number of images is reached. This ensures that the
    dataset is balanced across all classes and categories.
    Args:
        source_dir (str): The path to the source directory containing the
            dataset.
        augmentations (list): List of Albumentations transformations to apply.
        dataset_classes (list): Optional list of classes, e.g. ['COVID',
            'Lung_Opacity']. If None, all classes found in the source directory
            will be used.
        dataset_categories (list): Optional list of categories, e.g. ['images',
            'masks']. If None, all categories found in the source directory will
            be used.
        random_seed (int): Random seed for reproducibility.
    Returns:
        None
    """
    logger.info(f"> Augmenting dataset in {source_dir}.")
    if not path.exists(source_dir):
        logger.error(f"\t- The source directory {source_dir} does not exist.")
        raise ValueError(f"> The source directory {source_dir} does not exist.")
    if dataset_categories is None:
        dataset_categories = [
            d
            for d in os.listdir(source_dir)
            if path.isdir(path.join(source_dir, d))
        ]
    if dataset_classes is None:
        dataset_classes = [
            d
            for d in os.listdir(path.join(source_dir, dataset_categories[0]))
            if path.isdir(path.join(source_dir, dataset_categories[0], d))
        ]

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Albumentations Pipeline
    augmentation_pipeline = albumentations.Compose(augmentations)

    # Get the maximum number of images per class in each category for each split
    logger.info(
        "\t- Retrieve maximum class counts for each split and category."
    )
    max_counts = {
        category: {split: 0 for split in ["train", "val", "test"]}
        for category in dataset_categories
    }
    for category in dataset_categories:
        category_path = path.join(source_dir, category)
        if not path.exists(category_path):
            logger.warning(
                f"\t- Category {category} does not exist in source directory."
                " Skipping."
            )
            continue

        for split in ["train", "val", "test"]:
            split_path = path.join(category_path, split)
            if not path.exists(split_path):
                logger.warning(
                    f"\t- Split {split} does not exist in category {category}."
                    " Skipping."
                )
                continue

            for cls in dataset_classes:
                cls_path = path.join(split_path, cls)
                if not path.exists(cls_path):
                    logger.warning(
                        f"\t- Class {cls} does not exist in split {split} of"
                        f" category {category}. Skipping."
                    )
                    continue
                file_paths = [
                    f
                    for f in os.listdir(cls_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                count = len(file_paths)
                logger.info(
                    f"\t- Found {count} images for Class {cls}, Category"
                    f" {category}, Split {split}."
                )
                if count > max_counts[category][split]:
                    max_counts[category][split] = count
    logger.info(f"\t- Maximum counts per category and split: {max_counts}")

    # Augmentation and copying images
    logger.info("\t- Augmenting and copying images to reach maximum counts.")
    for category in dataset_categories:
        category_path = path.join(source_dir, category)
        if not path.exists(category_path):
            logger.warning(
                f"\t- Category {category} does not exist in source directory."
                " Skipping."
            )
            continue

        for split in ["train", "val", "test"]:
            split_path = path.join(category_path, split)
            if not path.exists(split_path):
                logger.warning(
                    f"\t- Split {split} does not exist in category {category}."
                    " Skipping."
                )
                continue

            for cls in dataset_classes:
                cls_path = path.join(split_path, cls)
                if not path.exists(cls_path):
                    logger.warning(
                        f"\t- Class {cls} does not exist in split {split} of"
                        f" category {category}. Skipping."
                    )
                    continue

                # Get current count of images
                file_paths = [
                    f
                    for f in os.listdir(cls_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                current_count = len(file_paths)
                max_count = max_counts[category][split]

                if current_count >= max_count:
                    logger.info(
                        f"\t- Class {cls}, Category {category}, Split"
                        f" {split} already has enough images. Skipping"
                        " augmentation."
                    )
                    continue

                # Augment images until the maximum count is reached
                while current_count < max_count:
                    for file_name in random.sample(
                        file_paths,
                        min(len(file_paths), max_count - current_count),
                    ):
                        src_file = path.join(cls_path, file_name)
                        image = cv2.imread(src_file)
                        augmented = augmentation_pipeline(image=image)
                        augmented_image = augmented["image"]

                        # Save the augmented image
                        base_name, ext = path.splitext(file_name)
                        augmented_file_name = (
                            f"{base_name}_aug_{current_count}{ext}"
                        )
                        dst_file = path.join(cls_path, augmented_file_name)
                        cv2.imwrite(dst_file, augmented_image)

                        current_count += 1

                logger.info(
                    f"\t- Augmented {current_count} images for Class {cls},"
                    f" Category {category}, Split {split}."
                )
    logger.info(f"\t- Dataset augmentation completed in {source_dir}.")
