# 🦠 COVID-19 Classification

This repository contains the continuation of the machine learning project conducted during the "Machine Learning Engineer" course at [DataScientest](https://datascientest.com). In some parts it is simplified to focus on the main concepts and does not include any work from the other course participants. Other parts are extended to demonstrate the application of advanced techniques.

This project works seamlessly in both **Google Colab** and **local environments**, with automatic environment detection and path configuration.

## 📚 Overview

This project uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) to classify chest X-ray images into four categories: COVID-19, Normal, Pneumonia, and Lung Opacity. The goal is to build a robust model that can accurately identify COVID-19 cases from X-ray images.

## 📂 Project Structure

- `data/`: Contains the original datasets downloaded from Kaggle
- `data_split/`: Contains the processed datasets split into train/validation/test
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training
- `src/`: Source code modules for data processing, utilities, and model architecture
  - `util/`: Utility modules for environment setup, dataset processing, and image operations
  - `scripts/`: Standalone scripts for data transformation

## 🚀 Quick Start

### Google Colab Setup

1. Upload the project to your Google Drive

2. Create a `.env` file in your project directory with the appropriate Google Drive paths (see `.env.example`)

3. Download the dataset from Kaggle and upload to your Google Drive

4. Open the notebooks in Google Colab and run!

### Local Environment Setup

1. **Install uv**: Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

2. **Download Dataset**: Get the dataset from the Kaggle [project page](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place folders and EXCEL-files in the `data/` directory

3. **Environment Configuration**: Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to set the correct paths if needed.

4. **Install Dependencies**:
   ```bash
   uv sync
   ```

5. **Setup Pre-commit Hooks**:
   ```bash
   uv run pre-commit install
   ```

6. **Process the Data** (Option 1 - Command Line):
   ```bash
   uv run python -m src.scripts.data_processing_pipeline
   ```

7. **Process the Data** (Option 2 - Notebook):
   Open and run `notebooks/2-pre_processing.ipynb` for an interactive data processing experience.

## 🔄 Data Processing Pipeline

The data processing pipeline includes:

1. **Environment Detection**: Automatically detects Google Colab vs local environment
2. **Dataset Restructuring**: Reorganizes the original dataset into train/validation/test splits (72/18/10) with stratified sampling
3. **Mask Application**: Creates masked images by applying corresponding masks to focus on relevant image regions

## 🧐 Development Tools

Pre-commit hooks are set up to ensure code quality and consistency. They include checks for formatting, linting, and type checking. To run the pre-commit hooks manually:

```bash
uv run pre-commit run --all-files
```

### 🧪 Testing

The project includes comprehensive unit testing for all utility modules to ensure code reliability and maintainability.

#### 🚀 Running Tests

**Quick test run:**
```bash
uv run pytest
```

**With verbose output:**
```bash
uv run pytest -v
```

**With coverage analysis:**
```bash
uv run pytest --cov=src --cov-report=html
```

#### 📊 Test Coverage

The test suite covers environment detection, dataset processing, and image operations with comprehensive mocking for external dependencies like OpenCV.

#### 🤖 Automated Testing

Tests run automatically via pre-commit hooks **only when merging to the main branch** to ensure fast development while maintaining code quality.

To install pre-commit hooks:
```bash
uv run pre-commit install
```