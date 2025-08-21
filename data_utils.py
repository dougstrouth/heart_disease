import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define feature lists based on the full set of features after harmonization
TARGET_COLUMN = 'heart_disease'
ALL_NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'bmi']
ALL_CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'smoking', 'diabetes']

def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None

def harmonize_datasets(df_synthetic, df_uci, verbose_output=False):
    """
    Harmonizes two heart disease datasets for combination.
    Aligns column names, re-encodes 'thal', handles unique features, and adds source tracking.
    """
    if df_synthetic is None or df_uci is None:
        print("Cannot harmonize: One or both DataFrames are None.")
        return None

    if verbose_output: print("\n--- Harmonizing Datasets ---")

    df_synthetic_harmonized = df_synthetic.copy()
    df_uci_harmonized = df_uci.copy()

    if 'thalch' in df_uci_harmonized.columns:
        df_uci_harmonized.rename(columns={'thalch': 'thalach'}, inplace=True)
        if verbose_output: print("Renamed 'thalch' to 'thalach' in UCI dataset.")

    if 'num' in df_uci_harmonized.columns:
        df_uci_harmonized.rename(columns={'num': 'heart_disease'}, inplace=True)
        if verbose_output: print("Renamed 'num' to 'heart_disease' in UCI dataset.")

    thal_mapping_uci_to_synthetic = {0: 3, 1: 6, 2: 7}
    if 'thal' in df_uci_harmonized.columns:
        df_uci_harmonized['thal'] = pd.to_numeric(df_uci_harmonized['thal'], errors='coerce')
        df_uci_harmonized['thal'] = df_uci_harmonized['thal'].map(thal_mapping_uci_to_synthetic)
        if verbose_output: print("Re-encoded 'thal' column in UCI dataset and handled unmapped values.")

    if 'source' not in df_synthetic_harmonized.columns:
        df_synthetic_harmonized['source'] = 'Synthetic'
    if verbose_output: print("Ensured 'source' column in synthetic dataset.")

    df_uci_harmonized['source'] = 'UCI'
    if verbose_output: print("Added 'source' column to UCI dataset.")

    unique_synthetic_features = ['smoking', 'diabetes', 'bmi']
    for feature in unique_synthetic_features:
        if feature not in df_uci_harmonized.columns:
            df_uci_harmonized[feature] = 0
            if verbose_output: print(f"Added and imputed '{feature}' with 0 for UCI dataset.")

    final_columns = df_synthetic_harmonized.columns.tolist()
    df_uci_harmonized = df_uci_harmonized[final_columns].copy()

    if verbose_output: print("Datasets harmonized successfully. Ready for concatenation.")
    return df_synthetic_harmonized, df_uci_harmonized

def combine_and_clean_data(df_synthetic, df_uci, verbose_output=False):
    """
    Combines harmonized datasets, handles NaNs, and binarizes the target variable.
    """
    df_synthetic_harmonized, df_uci_harmonized = harmonize_datasets(df_synthetic, df_uci, verbose_output)
    if df_synthetic_harmonized is None or df_uci_harmonized is None:
        return None

    combined_df = pd.concat([df_synthetic_harmonized, df_uci_harmonized], ignore_index=True)
    print(f"Combined dataset created with {len(combined_df)} rows.")

    initial_rows = len(combined_df)
    combined_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    rows_after_dropna = len(combined_df)
    print(f"Dropped {initial_rows - rows_after_dropna} rows with NaN in '{TARGET_COLUMN}'.")
    print(f"Combined dataset now has {rows_after_dropna} rows.")

    for col in ALL_CATEGORICAL_FEATURES:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str)
            combined_df[col] = combined_df[col].fillna('missing')
            if verbose_output: print(f"Converted '{col}' to string and filled NaNs.")

    combined_df[TARGET_COLUMN] = combined_df[TARGET_COLUMN].astype(int)
    combined_df[TARGET_COLUMN] = (combined_df[TARGET_COLUMN] > 0).astype(int)
    if verbose_output: print(f"Binarized '{TARGET_COLUMN}': values > 0 converted to 1.")

    return combined_df

def perform_eda(df, dataset_name, numerical_features, categorical_features, show_plots=False, verbose_output=False):
    """
    Performs basic Exploratory Data Analysis (EDA) on the DataFrame.
    Prints head, info, describe, missing values, and target distribution.
    Generates and displays basic plots.
    """
    if df is None:
        print(f"Cannot perform EDA: {dataset_name} DataFrame is None.")
        return

    if verbose_output:
        print(f"\n--- EDA for {dataset_name} ---")
        print("Head:")
        print(df.head())
        print("\nInfo:")
        print(df.info())
        print("\nDescription:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())

        target_column = 'heart_disease'
        if target_column in df.columns:
            print(f"\nTarget distribution ({target_column}):")
            print(df[target_column].value_counts(normalize=True))

    if show_plots:
        target_column = 'heart_disease'
        if target_column in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=target_column, data=df)
            plt.title(f'Distribution of {target_column} ({dataset_name})')
            plt.show()

    if numerical_features and all(col in df.columns for col in numerical_features):
        if show_plots:
            df[numerical_features].hist(bins=15, figsize=(15, 10))
            plt.suptitle(f'Histograms of Numerical Features ({dataset_name})')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    if categorical_features:
        for col in categorical_features:
            if col in df.columns:
                if show_plots:
                    plt.figure(figsize=(6, 4))
                    sns.countplot(x=col, data=df)
                    plt.title(f'Distribution of {col} ({dataset_name})')
                    plt.show()

def get_preprocessor(categorical_features, numerical_features):
    """
    Creates and returns a ColumnTransformer for preprocessing.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def get_preprocessed_data(df, target_column, categorical_features, numerical_features, cache_dir="cache", verbose_output=False):
    """
    Preprocesses the DataFrame, splits into X and y, and caches the results.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')

    features_for_preprocessing_numerical = [f for f in numerical_features if f in X.columns and f != 'source']
    features_for_preprocessing_categorical = [f for f in categorical_features if f in X.columns and f != 'source']

    preprocessor = get_preprocessor(features_for_preprocessing_categorical, features_for_preprocessing_numerical)

    os.makedirs(cache_dir, exist_ok=True)
    X_cache_path = os.path.join(cache_dir, "X_combined.joblib")
    y_cache_path = os.path.join(cache_dir, "y_combined.joblib")
    preprocessor_cache_path = os.path.join(cache_dir, "preprocessor_combined.joblib")

    if os.path.exists(X_cache_path) and os.path.exists(y_cache_path) and os.path.exists(preprocessor_cache_path):
        if verbose_output: print("\nLoading preprocessed data from cache...")
        X_cached = joblib.load(X_cache_path)
        y_cached = joblib.load(y_cache_path)
        preprocessor_cached = joblib.load(preprocessor_cache_path)
        if verbose_output: print("Preprocessed data loaded from cache.")
        return X_cached, y_cached, preprocessor_cached
    else:
        if verbose_output: print("\nPreprocessing data and saving to cache...")
        X_processed = preprocessor.fit_transform(X)
        
        joblib.dump(X_processed, X_cache_path)
        joblib.dump(y, y_cache_path)
        joblib.dump(preprocessor, preprocessor_cache_path)
        if verbose_output: print("Preprocessed data saved to cache.")
        return X_processed, y, preprocessor
