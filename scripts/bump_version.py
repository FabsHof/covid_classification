#!/usr/bin/env python3
"""
Simple version bumping script for COVID classification project.
Usage: python scripts/bump_version.py [major|minor|patch]
"""

import argparse
import re
import sys
from pathlib import Path


def get_current_version(pyproject_path: Path) -> str:
    """Extract current version from pyproject.toml."""
    content = pyproject_path.read_text()
    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def bump_version(version: str, bump_type: str) -> str:
    """Bump version according to semantic versioning."""
    major, minor, patch = map(int, version.split("."))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_version_in_file(file_path: Path, old_version: str, new_version: str):
    """Update version in a specific file."""
    content = file_path.read_text()
    updated_content = content.replace(f'"{old_version}"', f'"{new_version}"')
    updated_content = updated_content.replace(
        f"'{old_version}'", f"'{new_version}'"
    )
    file_path.write_text(updated_content)


def main():
    parser = argparse.ArgumentParser(description="Bump project version")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump",
    )
    args = parser.parse_args()

    # Project root
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    src_init_path = project_root / "src" / "__init__.py"

    # Get current version
    try:
        current_version = get_current_version(pyproject_path)
        print(f"📋 Current version: {current_version}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Calculate new version
    new_version = bump_version(current_version, args.bump_type)
    print(f"🚀 New version: {new_version}")

    # Update files
    try:
        update_version_in_file(pyproject_path, current_version, new_version)
        print(f"✅ Updated {pyproject_path}")

        if src_init_path.exists():
            update_version_in_file(src_init_path, current_version, new_version)
            print(f"✅ Updated {src_init_path}")

        print(
            f"🎉 Successfully bumped version from {current_version} to"
            f" {new_version}"
        )
        print(
            "💡 Don't forget to commit: git add . && git commit -m 'chore:"
            f" bump version to {new_version}'"
        )

    except Exception as e:
        print(f"❌ Error updating files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
