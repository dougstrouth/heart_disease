# Heart Disease Analysis Project

This project provides a comprehensive pipeline for analyzing the Heart Disease dataset. It includes data processing, model training, hyperparameter tuning, and experiment tracking with MLflow. The pipeline is designed to be modular and configurable, allowing for easy experimentation and iteration.

## Features

- **Data Processing**: Ingests and processes data from multiple sources, including the UCI Heart Disease dataset and synthetic data.
- **Model Training**: Trains and evaluates several classification models, including Logistic Regression, Random Forest, and XGBoost.
- **Hyperparameter Tuning**: Uses Optuna for intelligent hyperparameter optimization to find the best model configurations.
- **Experiment Tracking**: Leverages MLflow to log experiments, track model performance, and manage models in a model registry.
- **Orchestration**: A centralized `main.py` orchestrator to run different stages of the pipeline based on a configuration file.
- **Dask Integration**: Supports both local (Pandas) and distributed (Dask) computation for scalability.

## Data

This project uses a combination of datasets to analyze heart disease. The primary dataset is the UCI Heart Disease dataset, which contains 14 attributes from 76 attributes, including age, sex, chest pain type, resting blood pressure, cholesterol, and other clinical and lifestyle attributes. The goal is to predict the presence of heart disease.

In addition to the UCI dataset, this project also uses synthetic datasets for testing and development purposes. These synthetic datasets are designed to mimic the structure and characteristics of the original UCI dataset.

## Project Structure

```
├── .env.example
├── config.py
├── main.py
├── heart_disease_analysis.py
├── hyperparameter_tuning.py
├── review_experiments.py
├── modeling.py
├── model_training.py
├── preprocessing.py
├── data_schema.py
├── pyproject.toml
└── tests/
```

- **`main.py`**: The main entry point to the orchestrated pipeline.
- **`config.py`**: Configuration file for controlling the pipeline.
- **`heart_disease_analysis.py`**: Contains the main data analysis pipeline.
- **`hyperparameter_tuning.py`**: Script for running hyperparameter tuning with Optuna.
- **`review_experiments.py`**: Script for reviewing MLflow experiments and registering models.
- **`.env`**: File for storing environment variables (e.g., Databricks credentials).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Set up the Python environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    The project uses `pyproject.toml` to manage dependencies. Install them with pip:
    ```bash
    pip install -e .
    ```

## Environment Variables

This project requires certain environment variables to be set for connecting to the MLflow tracking server on Databricks. You should create a `.env` file in the root of the project directory.

Copy the `.env.example` file to `.env` and fill in your credentials:

```
# .env

# MLflow Configuration for Databricks
MLFLOW_TRACKING_URI=databricks

# Your Databricks workspace URL
DATABRICKS_HOST=https://<your-databricks-workspace-url>

# Your Databricks personal access token
DATABRICKS_TOKEN=<your-databricks-personal-access-token>
```

## Configuration

The `config.py` file is used to control the behavior of the pipeline. Here are the key configuration options:

### Orchestration

-   **`RUN_ANALYSIS`**: (boolean) If `True`, runs the main data analysis pipeline from `heart_disease_analysis.py`.
-   **`HYPERPARAMETER_TUNING_CONFIG`**: (dict) A dictionary to control hyperparameter tuning.
    -   `"run"`: (boolean) If `True`, runs the hyperparameter tuning script.
    -   `"cloud_run_mode"`: (string) Can be `"test"` or `"full"`. In `"test"` mode, it runs a single trial to verify functionality. `"full"` runs the complete tuning.
    -   `"coiled_max_time_hours"`: (int) The maximum time in hours for the Coiled run to complete.
-   **`RUN_EXPERIMENT_REVIEW`**: (boolean) If `True`, runs the experiment review script to find the best model and register it.

### General Configuration

-   **`DASK_TYPE`**: (string) Specifies the computation engine. Can be `'local'` (for Pandas), `'coiled'`, or `'cloud'` (for Dask).
-   **`SHOW_PLOTS`**: (boolean) If `True`, displays plots during EDA.
-   **`VERBOSE_OUTPUT`**: (boolean) If `True`, prints more detailed logs.

## Usage

The project is orchestrated through `main.py`, which reads the configuration from `config.py` to determine which steps to execute.

To run the pipeline, simply execute `main.py`:

```bash
python main.py
```

### Recommended Workflow

1.  **Run Hyperparameter Tuning**:
    -   In `config.py`, set `HYPERPARAMETER_TUNING_CONFIG["run"] = True` and the other orchestration flags to `False`.
    -   Configure `cloud_run_mode` and `coiled_max_time_hours` as needed.
    -   Run `python main.py`. This will execute the Optuna study and log all the experiment runs to MLflow.

2.  **Review Experiments and Register the Best Model**:
    -   Once the tuning is complete, change `config.py` to have `HYPERPARAMETER_TUNING_CONFIG["run"] = False` and `RUN_EXPERIMENT_REVIEW = True`.
    -   Run `python main.py` again. This will analyze the runs from the previous step, print a summary of the best one, and register the best model in the MLflow Model Registry.

3.  **Run the Main Analysis with the Best Model**:
    -   You can now update your analysis script to use the newly registered model for further analysis or inference.

## Scripts Overview

-   **`main.py`**: The central orchestrator for the entire pipeline.
-   **`heart_disease_analysis.py`**: A script that performs a full analysis of the heart disease data, including data loading, preprocessing, model training (with pre-defined parameters), and evaluation.
-   **`hyperparameter_tuning.py`**: A script dedicated to running hyperparameter optimization using Optuna. It systematically searches for the best model parameters and logs the results to MLflow.
-   **`review_experiments.py`**: A utility script to query MLflow, find the best performing model from the tuning runs, and register it in the MLflow Model Registry.
-   **`modeling.py`**: Contains the core logic for the model training pipeline, including data splitting, preprocessing, and model evaluation.
-   **`model_training.py`**: A lower-level script that handles the training and evaluation of a single model.
-   **`preprocessing.py`**: Contains functions for data preprocessing and feature engineering.
-   **`data_schema.py`**: Defines the schema for the heart disease dataset.
-   **`config.py`**: The central configuration file for the project.
-   **`.env`**: Stores sensitive information like API keys and credentials.
