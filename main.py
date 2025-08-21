import pandas as pd


def analyze_dataframe(df: pd.DataFrame, df_name: str):
    """
    Analyzes a pandas DataFrame, printing details about
    its dimensions, column names, data types, and null/blank value percentages.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        df_name (str): The name of the DataFrame to use in the output.
    """
    print("\n" + "="*50)
    print(f"--- Analyzing DataFrame: {df_name} ---")
    print("="*50 + "\n")

    # Get the dimensions (rows, columns)
    rows, cols = df.shape
    print(f"  Dimensions: {rows} rows, {cols} columns")

    if cols == 0:
        print("  The DataFrame has no columns to analyze.")
        return

    print("\n  Column Details:")
    # 1. Column Names
    print("    Column Names:")
    for col_name in df.columns:
        print(f"      - {col_name}")

    # 2. Data Types
    print("\n    Data Types:")
    for col_name, dtype in df.dtypes.items():
        print(f"      - {col_name}: {dtype}")

    # 3. Null/Blank Value Percentages
    print("\n    Null/Blank Value Percentages (includes NaNs and empty strings for text columns):")
    if rows > 0:
        for col_name in df.columns:
            nan_count = df[col_name].isnull().sum()
            
            empty_string_count = 0
            if df[col_name].dtype == 'object':
                empty_string_count = (df[col_name] == '').sum()
            
            total_problematic_values = nan_count + empty_string_count
            percentage = (total_problematic_values / rows) * 100
            print(f"      - {col_name}: {percentage:.2f}% missing/blank")
    else:
        print("      Cannot calculate percentages as there are no data rows.")

    # 4. Date/Time Fields (Improved detection)
    print("\n    Potential Date/Time Fields:")
    datetime_cols = []
    if rows > 0:
        for col_name in df.columns:
            try:
                temp_series = pd.to_datetime(df[col_name], errors='coerce')
                if temp_series.notna().sum() / rows > 0.8:
                     datetime_cols.append(col_name)
            except Exception:
                pass

    if datetime_cols:
        for col_name in datetime_cols:
            print(f"      - {col_name}")
        print("\n      Note: These columns *might* be date/time fields. Confirm with data exploration.")
        print("      Consider using the 'parse_dates' argument in pd.read_csv() when loading for analysis.")
    else:
        print("      No columns were automatically identified as potential date/time types.")
        print("      Consider using the 'parse_dates' argument in pd.read_csv() if you expect specific date columns.")

# --- Main execution ---
if __name__ == "__main__":
    # Path to the local CSV file
    file_path = '/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/pratyushpuri/heart-disease-dataset-3k-rows-python-code-2025/heart_disease_dataset.csv'
    
    try:
        # Load the dataset from the local CSV file
        df = pd.read_csv(file_path)
        
        # Analyze the DataFrame
        analyze_dataframe(df, "heart_disease_dataset.csv")

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
