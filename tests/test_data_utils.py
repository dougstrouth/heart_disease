import pandas as pd
import numpy as np
from data_utils import combine_and_clean_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES, TARGET_COLUMN

def test_pd_na_conversion_in_combine_and_clean_data():
    # Create a dummy DataFrame with pd.NA in various columns
    data_synthetic = {
        'age': [30, 40, pd.NA, 50],
        'sex': ['M', 'F', 'M', pd.NA],
        'cp': [0, 1, 2, pd.NA],
        'trestbps': [120, pd.NA, 130, 140],
        'chol': [200, 210, 220, pd.NA],
        'fbs': [0, 1, pd.NA, 0],
        'restecg': [0, 1, pd.NA, 0],
        'thalach': [150, 160, pd.NA, 170],
        'exang': [0, 1, 0, pd.NA],
        'oldpeak': [0.0, 1.0, pd.NA, 2.0],
        'slope': [0, 1, 2, pd.NA],
        'ca': [0, 1, pd.NA, 2],
        'thal': [0, 1, 2, pd.NA],
        'heart_disease': [0, 1, 0, 1],
        'smoking': [0, pd.NA, 1, 0],
        'diabetes': [1, 0, pd.NA, 1],
        'bmi': [25.0, 26.0, 27.0, pd.NA],
    }
    df_synthetic = pd.DataFrame(data_synthetic)

    # Create another dummy DataFrame (can be simpler as we're testing the cleaning of the first)
    data_uci = {
        'age': [35, 45, 55, 65],
        'sex': ['F', 'M', 'F', 'M'],
        'cp': [1, 2, 0, 1],
        'trestbps': [125, 135, 145, 155],
        'chol': [205, 215, 225, 235],
        'fbs': [1, 0, 1, 0],
        'restecg': [1, 0, 1, 0],
        'thalach': [155, 165, 175, 185],
        'exang': [1, 0, 1, 0],
        'oldpeak': [0.5, 1.5, 2.5, 3.5],
        'slope': [1, 2, 0, 1],
        'ca': [1, 2, 0, 1],
        'thal': [1, 2, 0, 1],
        'heart_disease': [1, 0, 1, 0],
        'smoking': [0, 0, 0, 0],
        'diabetes': [0, 0, 0, 0],
        'bmi': [28.0, 29.0, 30.0, 31.0],
    }
    df_uci = pd.DataFrame(data_uci)

    # Call the function under test
    combined_df = combine_and_clean_data(df_synthetic, df_uci, verbose_output=False)

    # Assertions
    assert combined_df is not None, "Combined DataFrame should not be None"

    # Check for pd.NA in all columns
    for col in combined_df.columns:
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            # For numeric columns, pd.NA should be converted to np.nan
            assert not combined_df[col].isna().any() or combined_df[col].isnull().all(), \
                f"Column {col} (numeric) still contains pd.NA or other missing values after cleaning."
        elif pd.api.types.is_object_dtype(combined_df[col]):
            # For object columns, pd.NA should be converted to np.nan or filled with 'missing'
            # Check if any element is literally pd.NA
            assert not (combined_df[col] == pd.NA).any(), \
                f"Column {col} (object) still contains literal pd.NA after cleaning."
            # Also check for np.nan if it's not supposed to be filled with 'missing'
            if col not in CATEGORICAL_FEATURES: # Categorical features are filled with 'missing'
                assert not combined_df[col].isna().any(), \
                    f"Column {col} (object) still contains np.nan after cleaning."

    # Verify that numerical features are numeric and categorical are object/string
    for col in NUMERICAL_FEATURES:
        if col in combined_df.columns:
            assert pd.api.types.is_numeric_dtype(combined_df[col]), f"Column {col} should be numeric."
    for col in CATEGORICAL_FEATURES:
        if col in combined_df.columns:
            assert pd.api.types.is_string_dtype(combined_df[col]) or pd.api.types.is_object_dtype(combined_df[col]), f"Column {col} should be string/object."
    for col in BINARY_FEATURES:
        if col in combined_df.columns:
            assert pd.api.types.is_numeric_dtype(combined_df[col]), f"Column {col} should be numeric."

    # Check that target column is binarized
    assert combined_df[TARGET_COLUMN].isin([0, 1]).all(), "Target column should only contain 0 or 1."