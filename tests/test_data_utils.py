import pytest
import pandas as pd
import os
import dask.dataframe as dd # Import dask.dataframe
from data_utils import load_data, harmonize_datasets, combine_and_clean_data # Import harmonize_datasets and combine_and_clean_data
from config import DASK_TYPE # Import DASK_TYPE

# Define the path to the dummy data relative to the project root
# This assumes the test is run from the project root or pytest is configured to find it.
dummy_data_path = os.path.join(os.path.dirname(__file__), "test_data", "dummy_data.csv")
synthetic_data_harmonize_path = os.path.join(os.path.dirname(__file__), "test_data", "synthetic_data_harmonize.csv")
uci_data_harmonize_path = os.path.join(os.path.dirname(__file__), "test_data", "uci_data_harmonize.csv")
dummy_data_with_string_in_numeric_path = os.path.join(os.path.dirname(__file__), "test_data", "dummy_data_with_string_in_numeric.csv")

def test_load_data_success():
    # Test successful loading of a CSV file
    df = load_data(dummy_data_path)
    assert df is not None

    if DASK_TYPE == 'coiled':
        assert isinstance(df, dd.DataFrame)
        # For Dask DataFrames, we might need to compute to check shape and columns
        # Or check dask-specific properties
        computed_df = df.compute()
        assert computed_df.shape == (3, 3)
        assert list(computed_df.columns) == ["col1", "col2", "col3"]
    else:
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)  # Check expected rows and columns
        assert list(df.columns) == ["col1", "col2", "col3"]

def test_load_data_file_not_found():
    # Test handling of a non-existent file
    non_existent_path = "/path/to/non_existent_file.csv"
    df = load_data(non_existent_path)
    assert df is None

def test_harmonize_datasets():
    df_synthetic = load_data(synthetic_data_harmonize_path)
    df_uci = load_data(uci_data_harmonize_path)

    df_synthetic_harmonized, df_uci_harmonized = harmonize_datasets(df_synthetic, df_uci)

    assert df_synthetic_harmonized is not None
    assert df_uci_harmonized is not None

    # Check if column names are consistent
    assert list(df_synthetic_harmonized.columns) == list(df_uci_harmonized.columns)

    # Check if 'thalch' was renamed to 'thalach' in UCI dataset
    assert 'thalach' in df_uci_harmonized.columns
    assert 'thalch' not in df_uci_harmonized.columns

    # Check if 'num' was renamed to 'heart_disease' in UCI dataset
    assert 'heart_disease' in df_uci_harmonized.columns
    assert 'num' not in df_uci_harmonized.columns

    # Check if 'thal' re-encoding is correct in UCI dataset
    # Original UCI thal: 0, 1, 2 -> Synthetic thal: 3, 6, 7
    # We need to compute if it's a Dask DataFrame
    if DASK_TYPE == 'coiled':
        uci_thal_values = df_uci_harmonized['thal'].compute().tolist()
    else:
        uci_thal_values = df_uci_harmonized['thal'].tolist()
    assert 3 in uci_thal_values
    assert 6 in uci_thal_values
    assert 7 in uci_thal_values
    assert 0 not in uci_thal_values # Original UCI value should be gone
    assert 1 not in uci_thal_values # Original UCI value should be gone
    assert 2 not in uci_thal_values # Original UCI value should be gone

    # Check if 'source' column was added
    assert 'source' in df_synthetic_harmonized.columns
    assert 'source' in df_uci_harmonized.columns

    # Check if unique synthetic features are handled (imputed with 0) in UCI dataset
    # These are 'smoking', 'diabetes', 'bmi'
    if DASK_TYPE == 'coiled':
        uci_smoking_values = df_uci_harmonized['smoking'].compute().tolist()
        uci_diabetes_values = df_uci_harmonized['diabetes'].compute().tolist()
        uci_bmi_values = df_uci_harmonized['bmi'].compute().tolist()
    else:
        uci_smoking_values = df_uci_harmonized['smoking'].tolist()
        uci_diabetes_values = df_uci_harmonized['diabetes'].tolist()
        uci_bmi_values = df_uci_harmonized['bmi'].tolist()

    # Assuming these columns were not in the original UCI data, they should be all 0s after imputation
    # For our dummy data, they are not present, so they should be imputed to 0.
    assert all(val == 0 for val in uci_smoking_values)
    assert all(val == 0 for val in uci_diabetes_values)
    assert all(val == 0 for val in uci_bmi_values)

def test_combine_and_clean_data_with_string_in_numeric_column():
    # Load the dummy data with a string in a numeric column
    df_with_string = load_data(dummy_data_with_string_in_numeric_path)
    # Create a dummy second dataframe for combine_and_clean_data
    df_dummy_second = pd.DataFrame({
        'age': [50], 'trestbps': [120], 'chol': [200], 'thalach': [150], 'oldpeak': [1.0], 'ca': [0],
        'bmi': [25], 'sex': [1], 'cp': [0], 'restecg': [0], 'slope': [0], 'thal': [3],
        'fbs': [0], 'exang': [0], 'smoking': [0], 'diabetes': [0], 'heart_disease': [1], 'source': ['Synthetic']
    })

    if DASK_TYPE == 'coiled':
        df_dummy_second = dd.from_pandas(df_dummy_second, npartitions=1)

    combined_df = combine_and_clean_data(df_with_string, df_dummy_second)

    assert combined_df is not None

    # Check if 'chol' column is numeric and contains NaN where the string was
    if DASK_TYPE == 'coiled':
        chol_values = combined_df['chol'].compute().tolist()
        heart_disease_values = combined_df['heart_disease'].compute().tolist()
    else:
        chol_values = combined_df['chol'].tolist()
        heart_disease_values = combined_df['heart_disease'].tolist()

    # Assert that the 'chol' column is now numeric (float) and contains NaN
    assert any(pd.isna(val) for val in chol_values)
    assert all(isinstance(val, (float, int)) for val in chol_values if not pd.isna(val))

    # Assert that heart_disease column is binarized (0 or 1)
    assert all(val in [0, 1] for val in heart_disease_values)
