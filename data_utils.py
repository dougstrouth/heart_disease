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
TARGET_COLUMN = 'heart_disease'
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'bmi']
CATEGORICAL_FEATURES = ['sex', 'cp', 'restecg', 'slope', 'thal']
BINARY_FEATURES = ['fbs', 'exang', 'smoking', 'diabetes']

def load_data(file_path):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Loads a CSV file into a pandas or Dask DataFrame based on DASK_TYPE.
    """
    try:
        if DASK_TYPE == 'coiled':
            df = dd.read_csv(file_path, dtype={'cp': 'object', 'restecg': 'object', 'sex': 'object', 'slope': 'object'})
            logger.info(f"Successfully loaded Dask DataFrame from {file_path}")
        else:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded pandas DataFrame from {file_path}")
        
        logger.debug(f"Head of {file_path} after loading:\n{df.head()}")
        logger.debug(f"Dtypes of {file_path} after loading:\n{df.dtypes}")
        
        return df
    except FileNotFoundError:
        logger.error(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
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

    if 'thalch' in df_uci_harmonized.columns:
        df_uci_harmonized = df_uci_harmonized.rename(columns={'thalch': 'thalach'})
        if verbose_output:
            logger.info("Renamed 'thalch' to 'thalach' in UCI dataset.")

    if 'num' in df_uci_harmonized.columns:
        df_uci_harmonized = df_uci_harmonized.rename(columns={'num': 'heart_disease'})
        if verbose_output:
            logger.info("Renamed 'num' to 'heart_disease' in UCI dataset.")

    thal_mapping_uci_to_synthetic = {0: 3, 1: 6, 2: 7}
    if 'thal' in df_uci_harmonized.columns:
        # Use apply for Dask DataFrames, map for Pandas
        if is_dask:
            df_uci_harmonized['thal'] = df_uci_harmonized['thal'].apply(lambda x: thal_mapping_uci_to_synthetic.get(x, np.nan), meta=('thal', 'float64'))
        else:
            df_uci_harmonized['thal'] = pd.to_numeric(df_uci_harmonized['thal'], errors='coerce')
            df_uci_harmonized['thal'] = df_uci_harmonized['thal'].map(thal_mapping_uci_to_synthetic)
        if verbose_output:
            logger.info("Re-encoded 'thal' column in UCI dataset and handled unmapped values.")

    if 'source' not in df_synthetic_harmonized.columns:
        if is_dask:
            df_synthetic_harmonized['source'] = 'Synthetic' # Dask will broadcast this
        else:
            df_synthetic_harmonized['source'] = 'Synthetic'
    if verbose_output:
        logger.info("Ensured 'source' column in synthetic dataset.")

    if is_dask:
        df_uci_harmonized['source'] = 'UCI' # Dask will broadcast this
    else:
        df_uci_harmonized['source'] = 'UCI'
    if verbose_output:
        logger.info("Added 'source' column to UCI dataset.")

    unique_synthetic_features = ['smoking', 'diabetes', 'bmi']
    for feature in unique_synthetic_features:
        if feature not in df_uci_harmonized.columns:
            if is_dask:
                df_uci_harmonized[feature] = 0 # Dask will broadcast this
            else:
                df_uci_harmonized[feature] = 0
            if verbose_output:
                logger.info(f"Added and imputed '{feature}' with 0 for UCI dataset.")

    # Define the expected final set of columns based on the overall schema
    expected_final_columns = list(set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [TARGET_COLUMN, 'source']))

    # For Dask, ensure all columns are present in both DFs before concatenation
    if is_dask:
        df_synthetic_harmonized = df_synthetic_harmonized.repartition(npartitions=df_uci_harmonized.npartitions) # Align partitions
        df_uci_harmonized = df_uci_harmonized.repartition(npartitions=df_synthetic_harmonized.npartitions) # Align partitions
        
        # Explicitly add missing columns with NaN for Dask DataFrames
        for col in expected_final_columns:
            if col not in df_synthetic_harmonized.columns:
                df_synthetic_harmonized[col] = np.nan
            if col not in df_uci_harmonized.columns:
                df_uci_harmonized[col] = np.nan

        # After adding missing columns, ensure binary features that might have been filled with NaN are set to 0
        for feature in BINARY_FEATURES:
            if feature in df_synthetic_harmonized.columns:
                df_synthetic_harmonized[feature] = df_synthetic_harmonized[feature].fillna(0)
            if feature in df_uci_harmonized.columns:
                df_uci_harmonized[feature] = df_uci_harmonized[feature].fillna(0)

    # Ensure the order of columns is consistent for concatenation
    df_synthetic_harmonized = df_synthetic_harmonized[expected_final_columns]
    df_uci_harmonized = df_uci_harmonized[expected_final_columns]

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
        # Dask's dropna requires meta for column types if not all columns are known
        combined_df = combined_df.dropna(subset=[TARGET_COLUMN])
    else:
        initial_rows = len(combined_df)
        combined_df.dropna(subset=[TARGET_COLUMN], inplace=True)
        rows_after_dropna = len(combined_df)
        logger.info(f"Dropped {initial_rows - rows_after_dropna} rows with NaN in '{TARGET_COLUMN}'.")
        logger.info(f"Combined dataset now has {rows_after_dropna} rows.")

    for col in CATEGORICAL_FEATURES:
        if col in combined_df.columns:
            if is_dask:
                # Dask's astype and fillna
                combined_df[col] = combined_df[col].astype(str)
                combined_df[col] = combined_df[col].fillna('missing')
            else:
                combined_df[col] = combined_df[col].astype(str)
                combined_df[col] = combined_df[col].fillna('missing')
            if verbose_output:
                logger.info(f"Converted '{col}' to string and filled NaNs.")

    for col in BINARY_FEATURES:
        if col in combined_df.columns:
            if is_dask:
                # Dask's to_numeric and fillna
                combined_df[col] = dd.to_numeric(combined_df[col], errors='coerce')
                combined_df[col] = combined_df[col].fillna(0) # Fill NaNs with 0 for binary features
            else:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                combined_df[col] = combined_df[col].fillna(0) # Fill NaNs with 0 for binary features
            if verbose_output:
                logger.info(f"Filled NaNs in binary feature '{col}' with 0.")

    for col in NUMERICAL_FEATURES:
        if col in combined_df.columns:
            if is_dask:
                combined_df[col] = dd.to_numeric(combined_df[col], errors='coerce')
            else:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            if verbose_output:
                logger.info(f"Coerced '{col}' to numeric, converting errors to NaN.")

    if is_dask:
        # For Dask, ensure target column is computed and binarized
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

        target_column = 'heart_disease'
        if target_column in df.columns:
            logger.info(f"\nTarget distribution ({target_column}):")
            if is_dask:
                logger.info(df[target_column].value_counts().compute())
            else:
                logger.info(df[target_column].value_counts(normalize=True))

    if show_plots:
        # For plotting, Dask DataFrames need to be computed first
        if is_dask:
            df_plot = df.compute()
        else:
            df_plot = df

        target_column = 'heart_disease'
        if target_column in df_plot.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=target_column, data=df_plot)
            plt.title(f'Distribution of {target_column} ({dataset_name})')
            plt.show()

    if numerical_features and all(col in df.columns for col in numerical_features):
        if show_plots:
            if is_dask:
                df_plot = df[numerical_features].compute()
            else:
                df_plot = df[numerical_features]
            df_plot.hist(bins=15, figsize=(15, 10))
            plt.suptitle(f'Histograms of Numerical Features ({dataset_name})')
            plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Fixed: changed list to tuple
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