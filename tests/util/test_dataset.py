"""
Unit tests for dataset utility functions.
"""

import os
import shutil

import pytest
from tests.fixtures.test_fixtures import (
    assert_directory_structure_exists,
    count_files_in_directory,
)

from src.util.dataset import restructure_dataset


class TestRestructureDataset:
    """Test restructure_dataset function."""

    def test_restructure_dataset_creates_structure(
        self, sample_dataset_structure
    ):
        """Test that restructure_dataset creates expected structure."""
        source_dir = sample_dataset_structure["source_dir"]
        target_dir = sample_dataset_structure["target_dir"]
        classes = sample_dataset_structure["classes"]
        categories = sample_dataset_structure["categories"]

        restructure_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            dataset_classes=classes,
            dataset_categories=categories,
            split_ratio=(0.6, 0.2, 0.2),
            random_seed=42,
        )

        # Check that target directory structure was created
        assert_directory_structure_exists(
            target_dir, {"categories": categories, "classes": classes}
        )

    def test_restructure_dataset_file_distribution(
        self, sample_dataset_structure
    ):
        """Test that files are distributed correctly across splits."""
        source_dir = sample_dataset_structure["source_dir"]
        target_dir = sample_dataset_structure["target_dir"]
        classes = sample_dataset_structure["classes"]
        categories = sample_dataset_structure["categories"]

        restructure_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            dataset_classes=classes,
            dataset_categories=categories,
            split_ratio=(0.6, 0.2, 0.2),
            random_seed=42,
        )

        # Check file counts for each class and category
        for cls in classes:
            for category in categories:
                # Count original files
                original_count = len(
                    sample_dataset_structure["files"][cls][category]
                )

                # Count distributed files
                train_count = count_files_in_directory(
                    os.path.join(target_dir, category, "train", cls)
                )
                val_count = count_files_in_directory(
                    os.path.join(target_dir, category, "val", cls)
                )
                test_count = count_files_in_directory(
                    os.path.join(target_dir, category, "test", cls)
                )

                # Total should match original count
                assert train_count + val_count + test_count == original_count

                # Train should have the most files (60%)
                assert train_count >= val_count
                assert train_count >= test_count

    def test_restructure_dataset_split_consistency(
        self, sample_dataset_structure
    ):
        """Test that the same files go to the same splits across categories."""
        source_dir = sample_dataset_structure["source_dir"]
        target_dir = sample_dataset_structure["target_dir"]
        classes = sample_dataset_structure["classes"]
        categories = sample_dataset_structure["categories"]

        restructure_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            dataset_classes=classes,
            dataset_categories=categories,
            split_ratio=(0.6, 0.2, 0.2),
            random_seed=42,
        )

        # Check that corresponding files are in the same splits
        for cls in classes:
            for split in ["train", "val", "test"]:
                # Get file lists for each category
                category_files = {}
                for category in categories:
                    split_dir = os.path.join(target_dir, category, split, cls)
                    if os.path.exists(split_dir):
                        category_files[category] = set(os.listdir(split_dir))

                # All categories should have the same files in the same split
                if len(category_files) > 1:
                    file_sets = list(category_files.values())
                    for i in range(1, len(file_sets)):
                        assert (
                            file_sets[0] == file_sets[i]
                        ), f"File mismatch in {cls} {split}"

    def test_restructure_dataset_clears_target_directory(
        self, sample_dataset_structure
    ):
        """Test that restructure_dataset clears existing target directory."""
        source_dir = sample_dataset_structure["source_dir"]
        target_dir = sample_dataset_structure["target_dir"]
        classes = sample_dataset_structure["classes"]
        categories = sample_dataset_structure["categories"]

        # Create some files in target directory first
        os.makedirs(target_dir, exist_ok=True)
        dummy_file = os.path.join(target_dir, "dummy.txt")
        with open(dummy_file, "w") as f:
            f.write("dummy content")

        assert os.path.exists(dummy_file)

        restructure_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            dataset_classes=classes,
            dataset_categories=categories,
            split_ratio=(0.6, 0.2, 0.2),
            random_seed=42,
        )

        # Dummy file should be gone
        assert not os.path.exists(dummy_file)

        # But proper structure should exist
        assert_directory_structure_exists(
            target_dir, {"categories": categories, "classes": classes}
        )

    def test_restructure_dataset_reproducibility(
        self, sample_dataset_structure
    ):
        """Test that restructure_dataset produces reproducible results."""
        source_dir = sample_dataset_structure["source_dir"]
        target_dir1 = sample_dataset_structure["target_dir"] + "_1"
        target_dir2 = sample_dataset_structure["target_dir"] + "_2"
        classes = sample_dataset_structure["classes"]
        categories = sample_dataset_structure["categories"]

        # Run restructuring twice with same seed
        for target_dir in [target_dir1, target_dir2]:
            restructure_dataset(
                source_dir=source_dir,
                target_dir=target_dir,
                dataset_classes=classes,
                dataset_categories=categories,
                split_ratio=(0.6, 0.2, 0.2),
                random_seed=42,
            )

        # Compare results
        for cls in classes:
            for category in categories:
                for split in ["train", "val", "test"]:
                    dir1 = os.path.join(target_dir1, category, split, cls)
                    dir2 = os.path.join(target_dir2, category, split, cls)

                    if os.path.exists(dir1) and os.path.exists(dir2):
                        files1 = set(os.listdir(dir1))
                        files2 = set(os.listdir(dir2))
                        assert (
                            files1 == files2
                        ), f"Mismatch in {cls}/{category}/{split}"

        # Clean up
        shutil.rmtree(target_dir1, ignore_errors=True)
        shutil.rmtree(target_dir2, ignore_errors=True)

    def test_restructure_dataset_missing_source_directory(self, temp_dir):
        """Test restructure_dataset raises error for missing source dir."""
        source_dir = os.path.join(temp_dir, "nonexistent")
        target_dir = os.path.join(temp_dir, "target")

        with pytest.raises(ValueError, match="does not exist"):
            restructure_dataset(
                source_dir=source_dir,
                target_dir=target_dir,
                dataset_classes=["COVID"],
                dataset_categories=["images"],
                split_ratio=(0.6, 0.2, 0.2),
                random_seed=42,
            )

    def test_restructure_dataset_auto_detect_classes_and_categories(
        self, sample_dataset_structure
    ):
        """Test restructure_dataset auto-detects classes and categories."""
        source_dir = sample_dataset_structure["source_dir"]
        target_dir = sample_dataset_structure["target_dir"]

        restructure_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            dataset_classes=None,  # Should auto-detect
            dataset_categories=None,  # Should auto-detect
            split_ratio=(0.6, 0.2, 0.2),
            random_seed=42,
        )

        # Should still create proper structure
        assert_directory_structure_exists(
            target_dir,
            {
                "categories": sample_dataset_structure["categories"],
                "classes": sample_dataset_structure["classes"],
            },
        )

    def test_restructure_dataset_split_ratio_validation(
        self, sample_dataset_structure
    ):
        """Test various split ratio scenarios."""
        source_dir = sample_dataset_structure["source_dir"]
        target_dir = sample_dataset_structure["target_dir"]
        classes = sample_dataset_structure["classes"]
        categories = sample_dataset_structure["categories"]

        # Test edge case: very small files count
        test_cases = [
            (0.8, 0.1, 0.1),  # 80/10/10 split
            (1.0, 0.0, 0.0),  # All train
            (0.0, 0.0, 1.0),  # All test
        ]

        for i, split_ratio in enumerate(test_cases):
            current_target = f"{target_dir}_{i}"

            restructure_dataset(
                source_dir=source_dir,
                target_dir=current_target,
                dataset_classes=classes,
                dataset_categories=categories,
                split_ratio=split_ratio,
                random_seed=42,
            )

            # Verify structure exists
            assert os.path.exists(current_target)

            # Clean up
            shutil.rmtree(current_target, ignore_errors=True)
