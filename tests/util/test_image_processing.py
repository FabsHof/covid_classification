"""
Unit tests for image processing utility functions.
"""

import os
from unittest.mock import patch

import numpy as np
import pytest
from tests.fixtures.test_fixtures import create_test_image, create_test_mask

from src.util.image_processing import apply_mask_to_image, create_masked_images


class TestApplyMaskToImage:
    """Test apply_mask_to_image function with proper cv2 mocking."""

    @patch("src.util.image_processing.cv2")
    def test_apply_mask_to_image_basic_success(self, mock_cv2, temp_dir):
        """Test successful mask application with mocked cv2."""
        # Create test paths
        image_path = os.path.join(temp_dir, "test_image.png")
        mask_path = os.path.join(temp_dir, "test_mask.png")

        # Mock cv2 functions
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_mask = np.ones((100, 100), dtype=np.uint8) * 255
        mock_processed_mask = np.ones((100, 100), dtype=np.uint8) * 255
        mock_result = np.ones((100, 100, 3), dtype=np.uint8) * 128

        mock_cv2.imread.side_effect = [mock_image, mock_mask]
        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.resize.return_value = mock_mask
        mock_cv2.threshold.return_value = (127, mock_processed_mask)
        mock_cv2.THRESH_BINARY = 0
        mock_cv2.INTER_NEAREST = 0
        mock_cv2.bitwise_and.return_value = mock_result

        # Test the function
        result = apply_mask_to_image(image_path, mask_path)

        # Verify cv2 calls
        assert mock_cv2.imread.call_count == 2
        mock_cv2.imread.assert_any_call(image_path)
        mock_cv2.imread.assert_any_call(mask_path, mock_cv2.IMREAD_GRAYSCALE)
        mock_cv2.threshold.assert_called_once()
        mock_cv2.bitwise_and.assert_called_once()

        # Verify result
        np.testing.assert_array_equal(result, mock_result)

    @patch("src.util.image_processing.cv2")
    def test_apply_mask_different_dims(self, mock_cv2, temp_dir):
        """Test mask application with different image/mask dimensions."""
        image_path = os.path.join(temp_dir, "test_image.png")
        mask_path = os.path.join(temp_dir, "test_mask.png")

        # Mock different sized image and mask
        mock_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        mock_mask = np.ones((100, 100), dtype=np.uint8) * 255
        mock_resized_mask = np.ones((200, 200), dtype=np.uint8) * 255
        mock_processed_mask = np.ones((200, 200), dtype=np.uint8) * 255
        mock_result = np.ones((200, 200, 3), dtype=np.uint8) * 128

        mock_cv2.imread.side_effect = [mock_image, mock_mask]
        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.resize.return_value = mock_resized_mask
        mock_cv2.threshold.return_value = (127, mock_processed_mask)
        mock_cv2.THRESH_BINARY = 0
        mock_cv2.INTER_NEAREST = 0
        mock_cv2.bitwise_and.return_value = mock_result

        result = apply_mask_to_image(image_path, mask_path)

        # Verify resize was called due to dimension mismatch
        mock_cv2.resize.assert_called_once_with(
            mock_mask, (200, 200), interpolation=mock_cv2.INTER_NEAREST
        )
        np.testing.assert_array_equal(result, mock_result)

    @patch("src.util.image_processing.cv2")
    def test_apply_mask_to_image_missing_image(self, mock_cv2):
        """Test error handling when image file cannot be loaded."""
        mock_cv2.imread.side_effect = [None, None]  # Both files missing
        mock_cv2.IMREAD_GRAYSCALE = 0

        with pytest.raises(ValueError, match="Could not load image"):
            apply_mask_to_image("nonexistent_image.png", "nonexistent_mask.png")

    @patch("src.util.image_processing.cv2")
    def test_apply_mask_to_image_missing_mask(self, mock_cv2):
        """Test error handling when mask file cannot be loaded."""
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_cv2.imread.side_effect = [
            mock_image,
            None,
        ]  # Image loads, mask doesn't
        mock_cv2.IMREAD_GRAYSCALE = 0

        with pytest.raises(ValueError, match="Could not load mask"):
            apply_mask_to_image("valid_image.png", "nonexistent_mask.png")


class TestCreateMaskedImages:
    """Test create_masked_images function with comprehensive mocking."""

    @patch("src.util.image_processing.cv2")
    def test_create_masked_images_success(self, mock_cv2, temp_dir):
        """Test successful masked images creation with mocked cv2."""
        # Setup mock cv2
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mock_mask = np.ones((100, 100), dtype=np.uint8) * 255
        mock_masked = np.ones((100, 100, 3), dtype=np.uint8) * 128

        mock_cv2.imread.side_effect = lambda path, *args: (
            mock_image if "images" in path else mock_mask
        )
        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.resize.return_value = mock_mask
        mock_cv2.threshold.return_value = (127, mock_mask)
        mock_cv2.THRESH_BINARY = 0
        mock_cv2.INTER_NEAREST = 0
        mock_cv2.bitwise_and.return_value = mock_masked
        mock_cv2.imwrite.return_value = True

        # Create test directory structure
        split_data_path = temp_dir
        images_dir = os.path.join(split_data_path, "images")
        masks_dir = os.path.join(split_data_path, "masks")

        # Create class directories within split directories
        for split in ["train"]:
            for cls in ["COVID"]:
                images_cls_dir = os.path.join(images_dir, split, cls)
                masks_cls_dir = os.path.join(masks_dir, split, cls)
                os.makedirs(images_cls_dir, exist_ok=True)
                os.makedirs(masks_cls_dir, exist_ok=True)

                # Create actual test files
                create_test_image(images_cls_dir, "COVID-1.png", (100, 100, 3))
                create_test_mask(masks_cls_dir, "COVID-1.png", (100, 100))

        # Test the function
        create_masked_images(split_data_path, ["COVID"], ["train"])

        # Verify masked images directory was created
        expected_masked_dir = os.path.join(split_data_path, "masked_images")
        assert os.path.exists(expected_masked_dir)

        # Verify cv2 functions were called
        assert mock_cv2.imread.call_count >= 2  # At least image + mask
        mock_cv2.imwrite.assert_called()

    @patch("src.util.image_processing.cv2")
    @patch("shutil.rmtree")
    def test_create_masked_images_clears_existing_target(
        self, mock_rmtree, mock_cv2, temp_dir
    ):
        """Test that existing masked_images directory is cleared."""
        # Setup basic mocks
        mock_cv2.imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.resize.return_value = np.ones((100, 100), dtype=np.uint8)
        mock_cv2.threshold.return_value = (
            127,
            np.ones((100, 100), dtype=np.uint8),
        )
        mock_cv2.bitwise_and.return_value = np.ones(
            (100, 100, 3), dtype=np.uint8
        )
        mock_cv2.imwrite.return_value = True

        split_data_path = temp_dir

        # Create directory structure
        images_dir = os.path.join(split_data_path, "images", "train", "COVID")
        masks_dir = os.path.join(split_data_path, "masks", "train", "COVID")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        create_test_image(images_dir, "COVID-1.png", (100, 100, 3))
        create_test_mask(masks_dir, "COVID-1.png", (100, 100))

        # Create existing target directory
        target_dir = os.path.join(split_data_path, "masked_images")
        os.makedirs(target_dir, exist_ok=True)

        create_masked_images(split_data_path, ["COVID"], ["train"])

        # Verify shutil.rmtree was called to clear existing directory
        mock_rmtree.assert_called_once_with(target_dir)

    def test_create_masked_images_missing_images_directory(self, temp_dir):
        """Test graceful handling when images directory doesn't exist."""
        split_data_path = temp_dir

        # Don't create images directory
        create_masked_images(split_data_path, ["COVID"])

        # Function should return early without crashing
        assert True  # Function handled missing directories gracefully

    def test_create_masked_images_missing_masks_directory(self, temp_dir):
        """Test graceful handling when masks directory doesn't exist."""
        split_data_path = temp_dir

        # Create only images directory, not masks
        images_dir = os.path.join(split_data_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        create_masked_images(split_data_path, ["COVID"])

        # Function should return early without crashing
        assert True

    @patch("src.util.image_processing.cv2")
    def test_create_masked_images_multiple_classes_and_splits(
        self, mock_cv2, temp_dir
    ):
        """Test processing multiple classes and splits."""
        # Setup comprehensive cv2 mocking
        mock_cv2.imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        mock_cv2.IMREAD_GRAYSCALE = 0
        mock_cv2.resize.return_value = np.ones((100, 100), dtype=np.uint8)
        mock_cv2.threshold.return_value = (
            127,
            np.ones((100, 100), dtype=np.uint8),
        )
        mock_cv2.bitwise_and.return_value = np.ones(
            (100, 100, 3), dtype=np.uint8
        )
        mock_cv2.imwrite.return_value = True

        split_data_path = temp_dir
        classes = ["COVID", "Normal"]
        splits = ["train", "val"]

        # Create structure for each class and split
        images_dir = os.path.join(split_data_path, "images")
        masks_dir = os.path.join(split_data_path, "masks")

        for cls in classes:
            for split in splits:
                images_cls_dir = os.path.join(images_dir, split, cls)
                masks_cls_dir = os.path.join(masks_dir, split, cls)
                os.makedirs(images_cls_dir, exist_ok=True)
                os.makedirs(masks_cls_dir, exist_ok=True)

                # Create test files for each class and split
                create_test_image(images_cls_dir, f"{cls}-1.png", (100, 100, 3))
                create_test_mask(masks_cls_dir, f"{cls}-1.png", (100, 100))

        create_masked_images(split_data_path, classes, splits)

        # Verify structure for each class and split combination
        for cls in classes:
            for split in splits:
                expected_masked_dir = os.path.join(
                    split_data_path, "masked_images", split, cls
                )
                assert os.path.exists(expected_masked_dir)

        # Verify cv2.imwrite was called for each file (4 total: 2×2)
        assert mock_cv2.imwrite.call_count == 4

    @patch("src.util.image_processing.cv2")
    def test_create_masked_images_missing_mask_file(self, mock_cv2, temp_dir):
        """Test handling when corresponding mask file is missing."""
        # Mock successful image loading but handle missing mask gracefully
        mock_cv2.imread.side_effect = lambda path, *args: (
            np.ones((100, 100, 3), dtype=np.uint8) if "images" in path else None
        )
        mock_cv2.IMREAD_GRAYSCALE = 0

        split_data_path = temp_dir

        # Create only image file, no corresponding mask
        images_dir = os.path.join(split_data_path, "images", "train", "COVID")
        masks_dir = os.path.join(split_data_path, "masks", "train", "COVID")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        create_test_image(images_dir, "COVID-1.png", (100, 100, 3))
        # Intentionally don't create mask file

        # Should not crash, just log warning and continue
        create_masked_images(split_data_path, ["COVID"], ["train"])

        # Verify directory structure was still created
        expected_masked_dir = os.path.join(
            split_data_path, "masked_images", "train", "COVID"
        )
        assert os.path.exists(expected_masked_dir)
