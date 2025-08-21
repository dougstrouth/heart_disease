import warnings
# Suppress the specific FutureWarning from scikit-learn's ColumnTransformer
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.compose._column_transformer")

import time
import os

# Import configuration
from config import SHOW_PLOTS, VERBOSE_OUTPUT, DASK_TYPE, RUN_PARAMETER_SEARCH, RUN_STACKED_ENSEMBLE, META_CLASSIFIER

# Import utility functions
from dask_utils import get_dask_client
from data_utils import load_data, combine_and_clean_data, perform_eda, get_preprocessed_data, TARGET_COLUMN, ALL_NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES
from model_training import train_evaluate_model, train_stacked_model
from model_interpretation import interpret_model
from utils.logging_utils import log_run_results

from sklearn.model_selection import train_test_split

def run_data_pipeline(dask_client, verbose_output, show_plots):
    """
    Executes the data loading, harmonization, EDA, and preprocessing steps.
    """
    print("\n--- Data Pipeline ---")
    start_time_load = time.time()
    df_synthetic = load_data('unified_heart_disease_dataset.csv')
    df_uci = load_data('/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/edwankarimsony/heart-disease-data/heart_disease_uci.csv')
    end_time_load = time.time()
    print(f"Data Loading completed in {end_time_load - start_time_load:.2f} seconds.")

    start_time_harmonize = time.time()
    combined_df = combine_and_clean_data(df_synthetic, df_uci, verbose_output=verbose_output)
    end_time_harmonize = time.time()
    print(f"Data Harmonization and Combination completed in {end_time_harmonize - start_time_harmonize:.2f} seconds.")

    start_time_eda = time.time()
    if combined_df is not None:
        perform_eda(combined_df, "Combined Dataset", ALL_NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES, show_plots=show_plots, verbose_output=verbose_output)
    end_time_eda = time.time()
    print(f"EDA completed in {end_time_eda - start_time_eda:.2f} seconds.")

    start_time_preprocess = time.time()
    X_combined, y_combined, preprocessor_combined = get_preprocessed_data(
        combined_df, TARGET_COLUMN, ALL_CATEGORICAL_FEATURES, ALL_NUMERICAL_FEATURES, verbose_output=verbose_output
    )
    end_time_preprocess = time.time()
    print(f"Data Preprocessing completed in {end_time_preprocess - start_time_preprocess:.2f} seconds.")

    return X_combined, y_combined, preprocessor_combined

def run_model_training_and_evaluation(X_combined, y_combined, preprocessor_combined, dask_client, verbose_output, run_stacked_ensemble, meta_classifier):
    """
    Handles model training, evaluation, and stacked ensemble if enabled.
    """
    print("\n--- Model Training and Evaluation ---")
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)
    if verbose_output: print("\nData split into training and testing sets.")

    # Logistic Regression
    lr_model_combined, _, _, lr_metrics_combined = train_evaluate_model(
        X_train, y_train, X_test, y_test, preprocessor_combined, model_type='logistic_regression', param_grid={'classifier__C': [1.0]}, dask_client=dask_client
    )

    # Random Forest
    rf_model_combined, _, _, rf_metrics_combined = train_evaluate_model(
        X_train, y_train, X_test, y_test, preprocessor_combined, model_type='random_forest', param_grid={'classifier__n_estimators': [100]}, dask_client=dask_client
    )

    # XGBoost
    xgb_model_combined, _, _, xgb_metrics_combined = train_evaluate_model(
        X_train, y_train, X_test, y_test, preprocessor_combined, model_type='xgboost', param_grid={'classifier__n_estimators': [100]}, dask_client=dask_client
    )

    # Stacked Ensemble
    stacked_model = None
    stacked_metrics = None
    if run_stacked_ensemble:
        base_models = {
            'lr': lr_model_combined,
            'rf': rf_model_combined,
            'xgb': xgb_model_combined
        }
        stacked_model, _, _, stacked_metrics = train_stacked_model(
            base_models, X_train, y_train, X_test, y_test, meta_classifier, dask_client
        )
    return lr_metrics_combined, rf_metrics_combined, xgb_metrics_combined, stacked_metrics, rf_model_combined

def run_model_interpretation(rf_model_combined, preprocessor_combined):
    """
    Performs model interpretation for the Random Forest model.
    """
    print("\n--- Model Interpretability ---")
    if rf_model_combined is not None:
        interpret_model(
            rf_model_combined, preprocessor_combined, ALL_NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES
        )

def log_analysis_results(start_time_total, dask_type, lr_metrics, rf_metrics, xgb_metrics, stacked_metrics, run_stacked_ensemble):
    """
    Logs the final analysis results.
    """
    run_details = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_runtime_seconds': time.time() - start_time_total,
        'dask_type': dask_type,
        'lr_accuracy': lr_metrics['accuracy'],
        'lr_roc_auc': lr_metrics['roc_auc'],
        'lr_best_params': str(lr_metrics['best_params']),
        'rf_accuracy': rf_metrics['accuracy'],
        'rf_roc_auc': rf_metrics['roc_auc'],
        'rf_best_params': str(rf_metrics['best_params']),
        'xgb_accuracy': xgb_metrics['accuracy'],
        'xgb_roc_auc': xgb_metrics['roc_auc'],
        'xgb_best_params': str(xgb_metrics['best_params']),
    }
    if run_stacked_ensemble and stacked_metrics is not None:
        run_details['stacked_accuracy'] = stacked_metrics['accuracy']
        run_details['stacked_roc_auc'] = stacked_metrics['roc_auc']
        run_details['stacked_best_params'] = str(stacked_metrics['best_params'] if 'best_params' in stacked_metrics else None)
    else:
        run_details['stacked_accuracy'] = None
        run_details['stacked_roc_auc'] = None
        run_details['stacked_best_params'] = None

    log_run_results(run_details)

def main():
    start_time_total = time.time()
    print("\n--- Starting Heart Disease Analysis ---")

    dask_client = None
    try:
        dask_client = get_dask_client(cluster_type=DASK_TYPE)
        print(f"Dask client created: {dask_client.dashboard_link}")

        X_combined, y_combined, preprocessor_combined = run_data_pipeline(dask_client, VERBOSE_OUTPUT, SHOW_PLOTS)

        if X_combined is not None:
            lr_metrics_combined, rf_metrics_combined, xgb_metrics_combined, stacked_metrics, rf_model_combined = \
                run_model_training_and_evaluation(X_combined, y_combined, preprocessor_combined, dask_client, VERBOSE_OUTPUT, RUN_STACKED_ENSEMBLE, META_CLASSIFIER)

            run_model_interpretation(rf_model_combined, preprocessor_combined)

            log_analysis_results(start_time_total, DASK_TYPE, lr_metrics_combined, rf_metrics_combined, xgb_metrics_combined, stacked_metrics, RUN_STACKED_ENSEMBLE)

    finally:
        if dask_client:
            print("Closing Dask client...")
            dask_client.close()
            print("Dask client closed.")

    print(f"\n--- Analysis Complete on Combined Dataset --- Total time: {time.time() - start_time_total:.2f} seconds.")
    print("Recommendations for further speed improvement:")
    print("1. GridSearchCV: The most time-consuming part is often hyperparameter tuning with GridSearchCV.")
    print("   - Reduce the size of parameter grids (fewer values per parameter).")
    print("   - Reduce the 'cv' (cross-validation) folds.")
    print("   - Consider RandomizedSearchCV instead of GridSearchCV for large search spaces.")
    print("2. Data Size: For very large datasets, consider sampling or using distributed computing frameworks (e.g., Dask).")
    print("3. Feature Engineering/Selection: Reduce the number of features if many are redundant or irrelevant.")
    print("4. Progress Bars: For long-running loops or processes, consider using 'tqdm' for visual progress indication.")
    print("   (e.g., 'from tqdm import tqdm' and wrap iterables like 'for item in tqdm(my_list):')")
    print("   However, integrating tqdm directly into GridSearchCV's internal process is more complex.")
    print("5. Caching: If you run the script multiple times with the same data, save preprocessed data (X, y) to disk.")
    print("   (e.g., using joblib.dump and joblib.load) to avoid re-running preprocessing.)")

# --- Main Execution Block ---
if __name__ == "__main__":
    main()