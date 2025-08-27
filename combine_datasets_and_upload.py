import time
import pandas as pd
import numpy as np
import dask.dataframe as dd

from utils.logger_config import setup_logging
from data_schema import HeartDiseaseSchema # Import HeartDiseaseSchema for dtypes
from pandas_data_utils import load_and_harmonize_uci, load_and_harmonize_synthetic, load_and_harmonize_johnsmith, combine_datasets, preprocess_data

def combine_and_save_all_datasets(output_csv_path="combined_heart_disease_dataset.csv"):
    logger = setup_logging()
    logger.info("--- Starting Data Combination and Local Save ---")
    start_time_total = time.time()

    # Define paths for the three datasets
    file_path_pratyushpuri = '/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/pratyushpuri/heart-disease-dataset-3k-rows-python-code-2025/heart_disease_dataset.csv'
    file_path_edwankarimsony = '/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/edwankarimsony/heart-disease-data/heart_disease_uci.csv'
    file_path_johnsmith88 = '/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/johnsmith88/heart-disease-data/heart.csv'

    # Prepare dtype mapping for pandas read_csv
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

    logger.info(f"Loading {file_path_pratyushpuri} with pandas...")
    try:
        df_pratyushpuri = pd.read_csv(file_path_pratyushpuri, dtype=schema_dtypes)
    except Exception as e:
        logger.error(f"Error loading {file_path_pratyushpuri}: {e}")
        return None

    logger.info(f"Loading {file_path_edwankarimsony} with pandas...")
    try:
        # For UCI dataset, handle 'thalch' and 'num' columns which are renamed later
        # Load them as float to handle potential NaNs, then let harmonization handle.
        uci_dtypes = schema_dtypes.copy()
        if 'thalach' in uci_dtypes: # thalach is the target name for thalch
            uci_dtypes['thalch'] = np.float64 # Load thalch as float
        if 'heart_disease' in uci_dtypes: # heart_disease is the target name for num
            uci_dtypes['num'] = np.float64 # Load num as float

        df_edwankarimsony = pd.read_csv(file_path_edwankarimsony, dtype=uci_dtypes)
    except Exception as e:
        logger.error(f"Error loading {file_path_edwankarimsony}: {e}")
        return None

    logger.info(f"Loading {file_path_johnsmith88} with pandas...")
    try:
        # For johnsmith88 dataset, handle 'target' column which is renamed later
        johnsmith88_dtypes = schema_dtypes.copy()
        if 'heart_disease' in johnsmith88_dtypes: # heart_disease is the target name for target
            johnsmith88_dtypes['target'] = np.float64 # Load target as float

        df_johnsmith88 = pd.read_csv(file_path_johnsmith88, dtype=johnsmith88_dtypes)
    except Exception as e:
        logger.error(f"Error loading {file_path_johnsmith88}: {e}")
        return None

    if any(df is None for df in [df_pratyushpuri, df_edwankarimsony, df_johnsmith88]):
        logger.error("One or more datasets failed to load. Exiting.")
        return None

    # Harmonize individual datasets
    df_pratyushpuri_harmonized = load_and_harmonize_synthetic(file_path_pratyushpuri)
    df_edwankarimsony_harmonized = load_and_harmonize_uci(file_path_edwankarimsony)
    df_johnsmith88_harmonized = load_and_harmonize_johnsmith(file_path_johnsmith88)

    if any(df is None for df in [df_pratyushpuri_harmonized, df_edwankarimsony_harmonized, df_johnsmith88_harmonized]):
        logger.error("One or more datasets failed to harmonize. Exiting.")
        return None

    logger.info("Combining harmonized datasets...")
    combined_df = combine_datasets(df_edwankarimsony_harmonized, df_pratyushpuri_harmonized, df_johnsmith88_harmonized)

    if combined_df is None:
        logger.error("Dataset combination failed. Exiting.")
        return None

    logger.info("Preprocessing combined dataset...")
    final_combined_df = preprocess_data(combined_df)

    if final_combined_df is None:
        logger.error("Preprocessing failed. Exiting.")
        return None

    logger.info(f"Final combined DataFrame has {len(final_combined_df)} rows.")

    # Save the combined DataFrame to a local CSV file
    # Ensure it's a pandas DataFrame before saving
    if isinstance(final_combined_df, dd.DataFrame):
        final_combined_df = final_combined_df.compute() # Convert Dask to pandas if it somehow became Dask

    logger.info(f"Saving combined pandas DataFrame to {output_csv_path}...")
    final_combined_df.to_csv(output_csv_path, index=False)

    logger.info(f"Combined dataset saved to {output_csv_path}")
    logger.info(f"--- Data Combination and Local Save Completed in {time.time() - start_time_total:.2f} seconds ---")

    return output_csv_path

if __name__ == "__main__":
    combined_csv_file = combine_and_save_all_datasets()
    if combined_csv_file:
        print(f"Successfully created combined CSV: {combined_csv_file}")
        print("Next step: Upload this file to GCS at gs://my-heart-disease-data-bucket/data/combined_heart_disease_dataset.csv")
    else:
        print("Failed to create combined CSV.")