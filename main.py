import kagglehub
import os
import pandas as pd
import shutil

def download_and_organize_dataset(dataset_name: str, path: str = 'data') -> str:
    """
    Downloads a dataset from Kaggle and organizes its files into a specified directory.

    Args:
        dataset_name (str): The name of the dataset on Kaggle (e.g., 'owner/dataset-name').
        path (str): The base directory where the dataset will be stored.

    Returns:
        str: The full path to the directory where the dataset files are located.
    """
    full_path = os.path.join(path, dataset_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {os.path.abspath(full_path)}")
    else:
        print(f"Directory already exists: {os.path.abspath(full_path)}")

    print(f"Attempting to download dataset: {dataset_name}...")
    try:
        data_loc = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to temporary location: {data_loc}")

        # Move files from the temporary download location to the organized directory
        for item in os.listdir(data_loc):
            s = os.path.join(data_loc, item)
            d = os.path.join(full_path, item)
            shutil.move(s, d)
        print(f"Dataset files successfully moved to: {os.path.abspath(full_path)}")
    except Exception as e:
        print(f"Error downloading or organizing dataset: {e}")
        return None
    return full_path

def analyze_csv_dimensions(directory_path: str):
    """
    Analyzes all CSV files within a given directory, printing details about
    their dimensions, column names, data types, and null/blank value percentages.

    Args:
        directory_path (str): The path to the directory containing the CSV files.
    """
    print("\n" + "="*50)
    print(f"--- Analyzing CSV Dimensions in: {os.path.abspath(directory_path)} ---")
    print("="*50 + "\n")

    found_csv_files = False
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for item_name in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item_name)
            if os.path.isfile(item_path) and item_name.lower().endswith('.csv'):
                found_csv_files = True
                print(f"\n--- Analyzing file: {item_name} ---")
                try:
                    # Read the CSV file into a pandas DataFrame
                    # infer_datetime_format is deprecated, use date_parser or parse_dates
                    df = pd.read_csv(item_path, low_memory=False) # Removed infer_datetime_format
                    
                    # Get the dimensions (rows, columns)
                    rows, cols = df.shape
                    print(f"  Dimensions: {rows} rows, {cols} columns")

                    if cols == 0:
                        print("  The CSV file has no columns to analyze.")
                        continue # Move to the next file

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
                            if df[col_name].dtype == 'object': # Check for empty strings only in object/text columns
                                empty_string_count = (df[col_name] == '').sum()
                            
                            total_problematic_values = nan_count + empty_string_count
                            percentage = (total_problematic_values / rows) * 100
                            print(f"      - {col_name}: {percentage:.2f}% missing/blank")
                    else:
                        print("      Cannot calculate percentages as there are no data rows.")

                    # 4. Date/Time Fields (Improved detection)
                    print("\n    Potential Date/Time Fields:")
                    datetime_cols = []
                    for col_name in df.columns:
                        # Attempt to convert to datetime to see if it's a date column
                        # Use errors='coerce' to turn unparseable dates into NaT
                        temp_series = pd.to_datetime(df[col_name], errors='coerce')
                        # If a significant portion of values converted successfully, consider it a date column
                        # You might adjust the threshold based on your data (e.g., 0.5 for 50%)
                        if temp_series.notna().sum() / rows > 0.8: # Example: if more than 80% are valid dates
                             datetime_cols.append(col_name)

                    if datetime_cols:
                        for col_name in datetime_cols:
                            print(f"      - {col_name}")
                        print("\n      Note: These columns *might* be date/time fields. Confirm with data exploration.")
                        print("      Consider using the 'parse_dates' argument in pd.read_csv() when loading for analysis.")
                    else:
                        print("      No columns were automatically identified as potential date/time types.")
                        print("      Consider using the 'parse_dates' argument in pd.read_csv() if you expect specific date columns.")


                except pd.errors.EmptyDataError:
                    print(f"  Warning: The file '{item_name}' is empty and could not be parsed.")
                except Exception as e:
                    print(f"  Error reading or processing file '{item_name}': {e}")
        if not found_csv_files:
            print(f"No CSV files found in the dataset directory: {os.path.abspath(directory_path)}")
    else:
        print(f"Dataset directory not found or is not a directory: {os.path.abspath(directory_path)}")

# --- Main execution ---
if __name__ == "__main__":
    dataset_name = 'pratyushpuri/heart-disease-dataset-3k-rows-python-code-2025'
    base_data_path = '/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data'

    # Step 1: Download and organize the dataset
    dataset_full_path = download_and_organize_dataset(dataset_name, base_data_path)

    if dataset_full_path:
        # Step 2: Analyze the downloaded CSV files for cleanliness
        analyze_csv_dimensions(dataset_full_path)