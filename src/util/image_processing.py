"""
Image processing utilities for applying masks and creating masked images.
"""

import logging
import os
import shutil
from os import path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_mask_to_image(image_path: str, mask_path: str) -> np.ndarray:
    """
    Applies a mask to an image.

    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the mask image

    Returns:
        np.ndarray: The masked image
    """
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    # Get image dimensions
    image_height, image_width = image.shape[:2]

    # Resize mask to match image dimensions if they differ
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(
            mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST
        )

    # Ensure mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply mask to each channel
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def create_masked_images(
    split_data_path: str, dataset_classes: List[str], splits: List[str] = None
) -> None:
    """
    Creates masked images for all images in the dataset by applying
    corresponding masks.

    Args:
        split_data_path (str): Path to the split dataset directory
        dataset_classes (List[str]): List of dataset classes
        splits (List[str], optional): List of splits to process.
            Defaults to ['train', 'val', 'test']
    """
    if splits is None:
        splits = ["train", "val", "test"]

    logger.info(f"Creating masked images in {split_data_path}")

    images_dir = path.join(split_data_path, "images")
    masks_dir = path.join(split_data_path, "masks")

    if not path.exists(images_dir):
        logger.error(f"Images directory {images_dir} does not exist")
        return

    if not path.exists(masks_dir):
        logger.error(f"Masks directory {masks_dir} does not exist")
        return

    # Create masked_images directory structure
    masked_images_dir = path.join(split_data_path, "masked_images")

    # Clear existing masked_images directory if it exists to ensure clean
    # processing
    if path.exists(masked_images_dir):
        logger.info(
            "\t- Removing existing masked images directory:"
            f" {masked_images_dir}"
        )
        shutil.rmtree(masked_images_dir)

    # Create fresh directory structure
    for split in splits:
        for cls in dataset_classes:
            cls_masked_dir = path.join(masked_images_dir, split, cls)
            os.makedirs(cls_masked_dir, exist_ok=True)

    # Process each split and class
    for split in splits:
        for cls in dataset_classes:
            images_cls_dir = path.join(images_dir, split, cls)
            masks_cls_dir = path.join(masks_dir, split, cls)
            masked_cls_dir = path.join(masked_images_dir, split, cls)

            if not path.exists(images_cls_dir):
                logger.warning(
                    f"Images directory {images_cls_dir} does not exist."
                    " Skipping."
                )
                continue

            if not path.exists(masks_cls_dir):
                logger.warning(
                    f"Masks directory {masks_cls_dir} does not exist. Skipping."
                )
                continue

            # Get all image files
            image_files = [
                f
                for f in os.listdir(images_cls_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            processed_count = 0
            for image_file in image_files:
                image_path = path.join(images_cls_dir, image_file)

                # Find corresponding mask file with exactly the same name
                mask_path = path.join(masks_cls_dir, image_file)

                if not path.exists(mask_path):
                    logger.warning(
                        f"No mask found for image {image_file} in class {cls},"
                        f" split {split}"
                    )
                    continue

                try:
                    # Apply mask and save masked image
                    masked_image = apply_mask_to_image(image_path, mask_path)
                    masked_image_path = path.join(masked_cls_dir, image_file)
                    cv2.imwrite(masked_image_path, masked_image)
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing {image_file}: {str(e)}")

            logger.info(
                f"Processed {processed_count} images for class {cls}, split"
                f"Processed {processed_count} images for class {cls}, split {split}"
            )

    logger.info("Masked images creation completed")
