import logging
import os
import random
import shutil
from os import path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _get_dataset_structure(
    source_dir: str,
    dataset_classes: List[str] = None,
    dataset_categories: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """Auto-detect dataset classes and categories if not provided."""
    if dataset_classes is None:
        dataset_classes = [
            d
            for d in os.listdir(source_dir)
            if path.isdir(path.join(source_dir, d))
        ]

    if dataset_categories is None:
        first_class_path = path.join(source_dir, dataset_classes[0])
        dataset_categories = [
            d
            for d in os.listdir(first_class_path)
            if path.isdir(path.join(first_class_path, d))
        ]

    return dataset_classes, dataset_categories


def _split_files(
    image_files: List[str], split_ratio: Tuple[float, float, float]
) -> Tuple[List[str], List[str], List[str]]:
    """Split files according to the given ratio."""
    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_test = n_total - n_train - n_val

    train_files = image_files[:n_train]
    val_files = image_files[n_train : n_train + n_val]
    test_files = image_files[n_train + n_val : n_train + n_val + n_test]

    return train_files, val_files, test_files


def _copy_files_for_category(
    source_dir: str,
    target_dir: str,
    cls: str,
    category: str,
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
) -> None:
    """Copy files for a specific category using the predetermined split."""
    category_path = path.join(source_dir, cls, category)
    if not path.exists(category_path):
        logger.warning(
            f"\t- Category {category} does not exist in class {cls}. Skipping."
        )
        return

    splits_and_files = [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]

    for split, files in splits_and_files:
        split_dir = path.join(target_dir, category, split, cls)
        for file in files:
            src_file = path.join(category_path, file)
            dst_file = path.join(split_dir, file)

            if path.exists(src_file):
                os.makedirs(path.dirname(dst_file), exist_ok=True)
                shutil.copy(src_file, dst_file)
            else:
                logger.warning(
                    f"\t- File {file} not found in category {category} for"
                    f" class {cls}"
                )


def _process_single_class(
    source_dir: str,
    target_dir: str,
    cls: str,
    dataset_categories: List[str],
    split_ratio: Tuple[float, float, float],
) -> None:
    """Process a single class: get files, split them, and copy to target."""
    cls_path = path.join(source_dir, cls)
    if not path.exists(cls_path):
        logger.warning(
            f"\t- Class {cls} does not exist in source directory. Skipping."
        )
        return

    # Get image files to determine the split
    images_category_path = path.join(source_dir, cls, "images")
    if not path.exists(images_category_path):
        logger.warning(
            f"\t- Images category does not exist in class {cls}. Skipping."
        )
        return

    image_files = [
        f
        for f in os.listdir(images_category_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        return

    # Split files
    train_files, val_files, test_files = _split_files(image_files, split_ratio)

    logger.info(
        f"\t- Class {cls}: {len(train_files)} train, {len(val_files)} val,"
        f" {len(test_files)} test files"
    )

    # Apply the same split to all categories
    for category in dataset_categories:
        _copy_files_for_category(
            source_dir,
            target_dir,
            cls,
            category,
            train_files,
            val_files,
            test_files,
        )


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

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Validate source directory
    if not path.exists(source_dir):
        logger.error(f"\t- > The source directory {source_dir} does not exist.")
        raise ValueError(f"> The source directory {source_dir} does not exist.")

    # Clear and create target directory
    if path.exists(target_dir):
        logger.info(f"\t- Removing existing target directory: {target_dir}")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Get dataset structure
    dataset_classes, dataset_categories = _get_dataset_structure(
        source_dir, dataset_classes, dataset_categories
    )

    # Create target directory structure
    for category in dataset_categories:
        category_path = path.join(target_dir, category)
        os.makedirs(category_path, exist_ok=True)
        for split in ["train", "val", "test"]:
            split_path = path.join(category_path, split)
            os.makedirs(split_path, exist_ok=True)

    # Process each class
    for cls in dataset_classes:
        _process_single_class(
            source_dir, target_dir, cls, dataset_categories, split_ratio
        )

    logger.info(
        f"\t- Dataset restructuring completed. Files copied to {target_dir}."
    )
