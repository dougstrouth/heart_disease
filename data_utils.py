import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import dask.dataframe as dd

from config import DASK_TYPE

# Define feature lists based on the full set of features after harmonization
from data_schema import HeartDiseaseSchema

TARGET_COLUMN = HeartDiseaseSchema.TARGET_COLUMN
NUMERICAL_FEATURES = HeartDiseaseSchema.NUMERICAL_COLUMNS
CATEGORICAL_FEATURES = HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE
# Note: BINARY_FEATURES are now part of CATEGORICAL_FEATURES in the schema,
# but I'll keep a separate list for them if they need special handling in combine_and_clean_data
# For now, I'll just use CATEGORICAL_FEATURES from the schema.
# I will need to adjust combine_and_clean_data to handle all categorical features uniformly.

def load_data(file_path):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Loads a CSV file into a pandas or Dask DataFrame based on DASK_TYPE.
    Performs initial column and dtype validation based on HeartDiseaseSchema.
    """
    try:
        # Prepare dtype mapping from schema
        schema_dtypes = {}
        for col, col_type in HeartDiseaseSchema.COLUMNS.items():
            if col_type == int:
                # Use pandas nullable integer type for robustness with missing values
                schema_dtypes[col] = pd.Int64Dtype()
            elif col_type == float:
                schema_dtypes[col] = np.float64
            else:
                # For other types (e.g., categorical represented as int), load as object initially
                schema_dtypes[col] = object

        if DASK_TYPE == 'coiled' or DASK_TYPE == 'cloud':
            # Dask's read_csv handles dtypes differently, especially for nullable integers.
            # It's often better to let Dask infer and then cast, or use 'object' for mixed types.
            # For simplicity and robustness, we'll load as object for categorical-like columns
            # and let Dask infer for numerical, then cast later.
            # Or, provide a simplified dtype map for Dask.
            dask_dtype_map = {col: 'object' if col in HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE else HeartDiseaseSchema.COLUMNS[col].__name__ for col in HeartDiseaseSchema.COLUMNS}
            # Ensure numerical columns are not 'object' if they are truly numerical
            for col in HeartDiseaseSchema.NUMERICAL_COLUMNS:
                dask_dtype_map[col] = 'float64' if HeartDiseaseSchema.COLUMNS[col] == float else 'int64'
            dask_dtype_map[HeartDiseaseSchema.TARGET_COLUMN] = 'int64'

            df = dd.read_csv(file_path, dtype=dask_dtype_map)
            logger.info(f"Successfully loaded Dask DataFrame from {file_path}")
        else:
            df = pd.read_csv(file_path, dtype=schema_dtypes)
            logger.info(f"Successfully loaded pandas DataFrame from {file_path}")

        # Validate columns present
        missing_cols = [col for col in HeartDiseaseSchema.COLUMNS if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in loaded data: {missing_cols}. Attempting to add with NaNs.")
            for col in missing_cols:
                if DASK_TYPE == 'coiled' or DASK_TYPE == 'cloud':
                    df[col] = np.nan # Dask handles NaN for new columns
                else:
                    df[col] = pd.NA if HeartDiseaseSchema.COLUMNS[col] == int else np.nan

        # Ensure all schema columns are present and in the correct order
        # This also handles cases where extra columns might be present in the CSV
        # by selecting only the schema-defined columns.
        df = df[list(HeartDiseaseSchema.COLUMNS.keys())]

        logger.debug(f"Head of {file_path} after loading:\n{df.head()}")
        logger.debug(f"Dtypes of {file_path} after loading:\n{df.dtypes}")

        return df
    except FileNotFoundError:
        logger.error(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}: {e}")
        return None


def harmonize_datasets(df_synthetic, df_uci, verbose_output=False):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Harmonizes two heart disease datasets for combination.
    Aligns column names, re-encodes 'thal', handles unique features, and adds source tracking.
    Handles both pandas and Dask DataFrames.
    """
    if df_synthetic is None or df_uci is None:
        logger.error("Cannot harmonize: One or both DataFrames are None.")
        return None

    if verbose_output:
        logger.info("\n--- Harmonizing Datasets ---")

    # Determine if we are working with Dask or Pandas DataFrames
    is_dask = isinstance(df_synthetic, dd.DataFrame)

    df_synthetic_harmonized = df_synthetic.copy()
    df_uci_harmonized = df_uci.copy()

    logger.debug(f"df_synthetic_harmonized head before renames:\n{df_synthetic_harmonized.head()}")
    logger.debug(f"df_uci_harmonized head before renames:\n{df_uci_harmonized.head()}")

    # Specific renames for UCI dataset
    if 'thalch' in df_uci_harmonized.columns:
        df_uci_harmonized = df_uci_harmonized.rename(columns={'thalch': 'thalach'})
        if verbose_output:
            logger.info("Renamed 'thalch' to 'thalach' in UCI dataset.")

    if 'num' in df_uci_harmonized.columns:
        df_uci_harmonized = df_uci_harmonized.rename(columns={'num': HeartDiseaseSchema.TARGET_COLUMN})
        if verbose_output:
            logger.info(f"Renamed 'num' to '{HeartDiseaseSchema.TARGET_COLUMN}' in UCI dataset.")

    # Re-encode 'thal' for UCI dataset to align with synthetic if necessary
    # This mapping is specific to how the UCI 'thal' values (0,1,2) are expected to map
    # to the synthetic dataset's 'thal' values (3,6,7) before string mapping.
    # If the schema's CATEGORICAL_MAPPINGS for 'thal' is sufficient, this can be removed.
    # For now, keeping it as it was in the original code.
    thal_mapping_uci_to_synthetic = {0: 3, 1: 6, 2: 7}
    if 'thal' in df_uci_harmonized.columns:
        if is_dask:
            df_uci_harmonized['thal'] = df_uci_harmonized['thal'].apply(lambda x: thal_mapping_uci_to_synthetic.get(x, np.nan), meta=('thal', 'float64'))
        else:
            df_uci_harmonized['thal'] = pd.to_numeric(df_uci_harmonized['thal'], errors='coerce')
            df_uci_harmonized['thal'] = df_uci_harmonized['thal'].map(thal_mapping_uci_to_synthetic)
        if verbose_output:
            logger.info("Re-encoded 'thal' column in UCI dataset and handled unmapped values.")

    # Add 'source' column
    if 'source' not in df_synthetic_harmonized.columns:
        if is_dask:
            df_synthetic_harmonized['source'] = 'Synthetic'
        else:
            df_synthetic_harmonized['source'] = 'Synthetic'
    if verbose_output:
        logger.info("Ensured 'source' column in synthetic dataset.")

    if is_dask:
        df_uci_harmonized['source'] = 'UCI'
    else:
        df_uci_harmonized['source'] = 'UCI'
    if verbose_output:
        logger.info("Added 'source' column to UCI dataset.")

    # Identify columns present in schema but potentially missing in one of the datasets
    # and add them with default values (e.g., 0 or NaN)
    schema_columns_with_source = list(HeartDiseaseSchema.COLUMNS.keys()) + ['source']
    
    for col in schema_columns_with_source:
        if col not in df_synthetic_harmonized.columns:
            if is_dask:
                df_synthetic_harmonized[col] = np.nan # Dask handles NaN for new columns
            else:
                df_synthetic_harmonized[col] = pd.NA if HeartDiseaseSchema.COLUMNS.get(col) == int else np.nan
            if verbose_output:
                logger.info(f"Added missing column '{col}' to synthetic dataset.")
        
        if col not in df_uci_harmonized.columns:
            if is_dask:
                df_uci_harmonized[col] = np.nan # Dask handles NaN for new columns
            else:
                df_uci_harmonized[col] = pd.NA if HeartDiseaseSchema.COLUMNS.get(col) == int else np.nan
            if verbose_output:
                logger.info(f"Added missing column '{col}' to UCI dataset.")

    # Ensure the order of columns is consistent for concatenation
    # Use HeartDiseaseSchema.COLUMN_ORDER and add 'source' to it
    final_column_order = HeartDiseaseSchema.COLUMN_ORDER + ['source']
    
    df_synthetic_harmonized = df_synthetic_harmonized[final_column_order]
    df_uci_harmonized = df_uci_harmonized[final_column_order]

    logger.debug(f"df_synthetic_harmonized head after harmonization:\n{df_synthetic_harmonized.head()}")
    logger.debug(f"df_uci_harmonized head after harmonization:\n{df_uci_harmonized.head()}")
    logger.debug(f"df_synthetic_harmonized dtypes after harmonization:\n{df_synthetic_harmonized.dtypes}")
    logger.debug(f"df_uci_harmonized dtypes after harmonization:\n{df_uci_harmonized.dtypes}")

    if verbose_output:
        logger.info("Datasets harmonized successfully. Ready for concatenation.")
    return df_synthetic_harmonized, df_uci_harmonized

def combine_and_clean_data(df_synthetic, df_uci, verbose_output=False):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Combines harmonized datasets, handles NaNs, and binarizes the target variable.
    Handles both pandas and Dask DataFrames.
    """
    df_synthetic_harmonized, df_uci_harmonized = harmonize_datasets(df_synthetic, df_uci, verbose_output)
    if df_synthetic_harmonized is None or df_uci_harmonized is None:
        return None

    is_dask = isinstance(df_synthetic_harmonized, dd.DataFrame)

    if is_dask:
        combined_df = dd.concat([df_synthetic_harmonized, df_uci_harmonized], ignore_index=True)
        logger.info("Combined Dask dataset created.")
    else:
        combined_df = pd.concat([df_synthetic_harmonized, df_uci_harmonized], ignore_index=True)
        logger.info(f"Combined pandas dataset created with {len(combined_df)} rows.")

    logger.debug(f"Combined_df head before NaN drop:\n{combined_df.head()}")
    logger.debug(f"Combined_df dtypes before NaN drop:\n{combined_df.dtypes}")

    # Drop rows with NaN in TARGET_COLUMN
    if is_dask:
        combined_df = combined_df.dropna(subset=[TARGET_COLUMN])
    else:
        initial_rows = len(combined_df)
        combined_df.dropna(subset=[TARGET_COLUMN], inplace=True)
        rows_after_dropna = len(combined_df)
        logger.info(f"Dropped {initial_rows - rows_after_dropna} rows with NaN in '{TARGET_COLUMN}'.")
        logger.info(f"Combined dataset now has {rows_after_dropna} rows.")

    # Handle categorical features
    for col in HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE:
        if col in combined_df.columns:
            if is_dask:
                # Convert pd.NA to np.nan first, then fill np.nan with 'missing' or 0 for binary
                combined_df[col] = combined_df[col].replace(pd.NA, np.nan)
                # For binary features (0/1), fill NaNs with 0. Otherwise, fill with 'missing'.
                if col in ["fbs", "exang", "smoking", "diabetes"]: # These are the binary features
                    combined_df[col] = dd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)
                else:
                    combined_df[col] = combined_df[col].fillna('missing').astype(str)
            else:
                combined_df[col] = combined_df[col].replace(pd.NA, np.nan)
                if col in ["fbs", "exang", "smoking", "diabetes"]:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)
                else:
                    combined_df[col] = combined_df[col].fillna('missing').astype(str)
            if verbose_output:
                logger.info(f"Processed categorical feature '{col}'.")

    # Handle numerical features
    for col in HeartDiseaseSchema.NUMERICAL_COLUMNS:
        if col in combined_df.columns:
            if is_dask:
                combined_df[col] = combined_df[col].replace(pd.NA, np.nan)
                combined_df[col] = dd.to_numeric(combined_df[col], errors='coerce').fillna(0) # Fill NaNs with 0 for numerical features
                combined_df[col] = combined_df[col].astype(float) # Ensure float type
            else:
                combined_df[col] = combined_df[col].replace(pd.NA, np.nan)
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
                combined_df[col] = combined_df[col].astype(float)
            if verbose_output:
                logger.info(f"Processed numerical feature '{col}'.")

    # Ensure target column is binarized and correct type
    if is_dask:
        combined_df[TARGET_COLUMN] = combined_df[TARGET_COLUMN].astype(int)
        combined_df[TARGET_COLUMN] = (combined_df[TARGET_COLUMN] > 0).astype(int)
    else:
        combined_df[TARGET_COLUMN] = combined_df[TARGET_COLUMN].astype(int)
        combined_df[TARGET_COLUMN] = (combined_df[TARGET_COLUMN] > 0).astype(int)
    if verbose_output:
        logger.info(f"Binarized '{TARGET_COLUMN}': values > 0 converted to 1.")

    logger.debug(f"Combined_df head after cleaning:\n{combined_df.head()}")
    logger.debug(f"Combined_df dtypes after cleaning:\n{combined_df.dtypes}")

    return combined_df


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


def preprocess_data(df, preprocessor, target_column, cache_dir="cache", use_cache=True, verbose_output=False):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Preprocesses the DataFrame using a provided preprocessor, splits into X and y, and caches the results.
    Handles both pandas and Dask DataFrames.
    """
    is_dask = isinstance(df, dd.DataFrame)

    if is_dask:
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
    else:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

    if 'source' in X.columns:
        X = X.drop('source', axis=1)

    os.makedirs(cache_dir, exist_ok=True)
    X_cache_path = os.path.join(cache_dir, "X_processed.joblib")
    y_cache_path = os.path.join(cache_dir, "y_processed.joblib")

    if use_cache and os.path.exists(X_cache_path) and os.path.exists(y_cache_path):
        if verbose_output:
            logger.info("\nLoading preprocessed data from cache...")
        X_processed = joblib.load(X_cache_path)
        y_processed = joblib.load(y_cache_path)
        if verbose_output:
            logger.info("Preprocessed data loaded from cache.")
        return X_processed, y_processed

    if verbose_output:
        logger.info("\nPreprocessing data...")
    
    # Debugging: Check for NaNs before preprocessing
    if is_dask:
        # Dask's isnull().sum() returns a Series, need to compute sum of sums
        if X.isnull().sum().sum().compute() > 0:
            logger.warning("WARNING: NaNs found in X before preprocessing:")
            logger.warning(X.isnull().sum().compute()[X.isnull().sum().compute() > 0])
    else:
        if X.isnull().sum().sum() > 0:
            logger.warning("WARNING: NaNs found in X before preprocessing:")
            logger.warning(X.isnull().sum()[X.isnull().sum() > 0])
        
    X_processed = preprocessor.fit_transform(X)

    if use_cache:
        # For Dask DataFrames, compute before saving to joblib
        if is_dask:
            joblib.dump(X_processed.compute(), X_cache_path)
            joblib.dump(y.compute(), y_cache_path)
        else:
            joblib.dump(X_processed, X_cache_path)
            joblib.dump(y, y_cache_path)
        if verbose_output:
            logger.info("Preprocessed data saved to cache.")

    return X_processed, y