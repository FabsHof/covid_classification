# 🦠 COVID-19 Classification

This repository contains the continuation of the machine learning project conducted during the "Machine Learning Engineer" course at [DataScientest](https://datascientest.com). In some parts it is simplified to focus on the main concepts and does not include any work from the other course participants. Other parts are extended to demonstrate the application of advanced techniques.

## 📚 Overview

This project uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) to classify chest X-ray images into four categories: COVID-19, Normal, Pneumonia, and Lung Opacity. The goal is to build a robust model that can accurately identify COVID-19 cases from X-ray images.

## 📂 Project Structure

- `data/`: Contains the datasets used for training and testing.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `src/`: Source code for the model architecture, training, and evaluation.

## 👟 Setup

1. Install `uv` from the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. Download the dataset from the Kaggle [project page](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place folders and EXCEL-files in the `data/` directory.
3. Create and edit the `.env`-file to set the correct paths (see `.env.example` for reference).
4. Install pre-commit hooks by running:
   ```bash
   uv run pre-commit install
   ```
5. Run the following command to transform, split and augment the data:
   ```bash
   uv run python -m src.scripts.data_transform
   ```

### 🧐 Pre-Commit Hooks

Pre-commit hooks are set up to ensure code quality and consistency. They include checks for formatting, linting, and type checking. To run the pre-commit hooks manually, use:
```bash
uv run pre-commit run --all-files
```