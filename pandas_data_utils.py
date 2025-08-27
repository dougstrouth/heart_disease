
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import dask.dataframe as dd
from data_schema import HeartDiseaseSchema

TARGET_COLUMN = HeartDiseaseSchema.TARGET_COLUMN

def load_and_harmonize_uci(file_path, schema_dtypes):
    """Loads and harmonizes the UCI heart disease dataset."""
    df = pd.read_csv(file_path, na_values=['?'], dtype=schema_dtypes)
    df = df.rename(columns={'num': 'target'})
    df['origin'] = 'uci'
    return df

def load_and_harmonize_synthetic(file_path, schema_dtypes):
    """Loads and harmonizes the synthetic heart disease dataset."""
    df = pd.read_csv(file_path, na_values=['?'], dtype=schema_dtypes)
    df = df.rename(columns={'heart_disease': 'target'})
    df['origin'] = 'synthetic'
    return df

def load_and_harmonize_johnsmith(file_path, schema_dtypes):
    """Loads and harmonizes the John Smith heart disease dataset."""
    df = pd.read_csv(file_path, na_values=['?'], dtype=schema_dtypes)
    # No target column rename needed for this dataset as it's already named 'target'
    df['origin'] = 'johnsmith'
    return df

def combine_datasets(uci_df, synthetic_df, johnsmith_df):
    """Combines the three harmonized datasets."""
    combined_df = pd.concat([uci_df, synthetic_df, johnsmith_df], ignore_index=True)
    return combined_df

def preprocess_data(df):
    """Applies preprocessing steps to the combined dataset."""
    # Harmonize 'thal' column
    df['thal'] = df['thal'].replace({0: 'normal', 1: 'fixed defect', 2: 'reversable defect', 3: 'normal', 6: 'fixed defect', 7: 'reversable defect'})

    # Harmonize 'sex'
    df['sex'] = df['sex'].replace({0: 'female', 1: 'male'})

    # Harmonize 'cp'
    df['cp'] = df['cp'].replace({0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal', 3: 'asymptomatic', 4: 'asymptomatic'})

    # Harmonize 'restecg'
    df['restecg'] = df['restecg'].replace({0: 'normal', 1: 'stt abnormality', 2: 'lv hypertrophy'})

    # Convert target to binary
    df['target'] = np.where(df['target'] > 0, 1, 0)
    df = df.rename(columns={'target': 'heart_disease'})

    return df

def get_processed_data(uci_path, synthetic_path, johnsmith_path):
    """Main function to load, harmonize, and preprocess the datasets."""
    schema_dtypes = {}
    for col, col_type in HeartDiseaseSchema.COLUMNS.items():
        if col in HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE():
            schema_dtypes[col] = object # Load as object if it's a categorical column
        elif isinstance(col_type, type) and issubclass(col_type, int):
            schema_dtypes[col] = pd.Int64Dtype() # Use nullable integer type for non-categorical ints
        elif isinstance(col_type, type) and issubclass(col_type, float):
            schema_dtypes[col] = np.float64
        else:
            schema_dtypes[col] = object

    uci_df = load_and_harmonize_uci(uci_path, schema_dtypes)
    synthetic_df = load_and_harmonize_synthetic(synthetic_path, schema_dtypes)
    johnsmith_df = load_and_harmonize_johnsmith(johnsmith_path, schema_dtypes)

    combined_df = combine_datasets(uci_df, synthetic_df, johnsmith_df)
    processed_df = preprocess_data(combined_df)

    return processed_df

def perform_eda(df, dataset_name, numerical_features, categorical_features, show_plots=False, verbose_output=False):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Performs basic Exploratory Data Analysis (EDA) on the DataFrame.
    Handles both pandas and Dask DataFrames.
    """
    is_dask = isinstance(df, dd.DataFrame)

    if df is None:
        logger.error(f"Cannot perform EDA: {dataset_name} DataFrame is None.")
        return

    if verbose_output:
        logger.info(f"\n--- EDA for {dataset_name} ---")
        logger.info("Head:")
        if is_dask:
            logger.info(df.head().compute())
        else:
            logger.info(df.head())
        logger.info("\nInfo:")
        # Dask DataFrames do not have a direct .info() equivalent that prints to buffer
        if is_dask:
            logger.info(df.describe().compute())
        else:
            df.info(buf=logger.info) # Redirect info to logger
        logger.info("\nDescription:")
        if is_dask:
            logger.info(df.describe().compute())
        else:
            logger.info(df.describe())
        logger.info("\nMissing values:")
        if is_dask:
            logger.info(df.isnull().sum().compute())
        else:
            logger.info(df.isnull().sum())

        # Use the global TARGET_COLUMN from schema
        if TARGET_COLUMN in df.columns:
            logger.info(f"\nTarget distribution ({TARGET_COLUMN}):")
            if is_dask:
                logger.info(df[TARGET_COLUMN].value_counts().compute())
            else:
                logger.info(df[TARGET_COLUMN].value_counts(normalize=True))

    if show_plots:
        # For plotting, Dask DataFrames need to be computed first
        if is_dask:
            df_plot = df.compute()
        else:
            df_plot = df

        # Use the global TARGET_COLUMN from schema
        if TARGET_COLUMN in df_plot.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=TARGET_COLUMN, data=df_plot)
            plt.title(f'Distribution of {TARGET_COLUMN} ({dataset_name})')
            plt.show()

    if numerical_features and all(col in df.columns for col in numerical_features):
        if show_plots:
            if is_dask:
                df_plot = df[numerical_features].compute()
            else:
                df_plot = df[numerical_features]
            df_plot.hist(bins=15, figsize=(15, 10))
            plt.suptitle(f'Histograms of Numerical Features ({dataset_name})')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.show()

    if categorical_features:
        for col in categorical_features:
            if col in df.columns:
                if show_plots:
                    if is_dask:
                        df_plot = df[[col]].compute()
                    else:
                        df_plot = df[[col]]
                    plt.figure(figsize=(6, 4))
                    sns.countplot(x=col, data=df_plot)
                    plt.title(f'Distribution of {col} ({dataset_name})')
                    plt.show()
