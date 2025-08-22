
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.compose._column_transformer")

import time
import pandas as pd

from config import SHOW_PLOTS, VERBOSE_OUTPUT, DASK_TYPE, RUN_STACKED_ENSEMBLE, META_CLASSIFIER
from dask_utils import get_dask_client
from data_utils import load_data, combine_and_clean_data, perform_eda, preprocess_data, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
from model_training import train_evaluate_model, train_stacked_model
from model_interpretation import interpret_model
from preprocessing import get_preprocessor, get_feature_names
from utils.logging_utils import log_run_results

from sklearn.model_selection import train_test_split

def run_data_pipeline(verbose_output, show_plots):
    print("\n--- Data Pipeline ---")
    start_time_load = time.time()
    df_synthetic = load_data('unified_heart_disease_dataset.csv')
    df_uci = load_data('/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/edwankarimsony/heart-disease-data/heart_disease_uci.csv')
    print(f"Data Loading completed in {time.time() - start_time_load:.2f} seconds.")

    start_time_harmonize = time.time()
    combined_df = combine_and_clean_data(df_synthetic, df_uci, verbose_output=verbose_output)
    print(f"Data Harmonization and Combination completed in {time.time() - start_time_harmonize:.2f} seconds.")

    if combined_df is not None:
        perform_eda(combined_df, "Combined Dataset", NUMERICAL_FEATURES, CATEGORICAL_FEATURES, show_plots=show_plots, verbose_output=verbose_output)

    return combined_df

def run_model_pipeline(dask_client, combined_df, verbose_output, run_stacked_ensemble, meta_classifier):
    print("\n--- Model Training, Evaluation, and Interpretation ---")

    X = combined_df.drop(TARGET_COLUMN, axis=1)
    y = combined_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if verbose_output: print("\nData split into training and testing sets.")

    preprocessor = get_preprocessor(CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    # Logistic Regression
    lr_model, _, _, lr_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, model_type='logistic_regression', param_grid={'classifier__C': [1.0]}, dask_client=dask_client)

    # Random Forest
    rf_model, _, _, rf_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, model_type='random_forest', param_grid={'classifier__n_estimators': [100]}, dask_client=dask_client)

    # XGBoost
    xgb_model, _, _, xgb_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, model_type='xgboost', param_grid={'classifier__n_estimators': [100]}, dask_client=dask_client)

    stacked_metrics = None
    if run_stacked_ensemble:
        base_models = {'lr': lr_model, 'rf': rf_model, 'xgb': xgb_model}
        _, _, _, stacked_metrics = train_stacked_model(base_models, X_train_processed, y_train, X_test_processed, y_test, meta_classifier, dask_client)

    print("\n--- Model Interpretability (Random Forest) ---")
    if rf_model is not None:
        interpret_model(rf_model, X_train_processed, feature_names)

    return lr_metrics, rf_metrics, xgb_metrics, stacked_metrics

def log_analysis_results(start_time, dask_type, lr_metrics, rf_metrics, xgb_metrics, stacked_metrics, run_stacked_ensemble):
    run_details = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_runtime_seconds': time.time() - start_time,
        'dask_type': dask_type,
        'lr_accuracy': lr_metrics['accuracy'], 'lr_roc_auc': lr_metrics['roc_auc'],
        'rf_accuracy': rf_metrics['accuracy'], 'rf_roc_auc': rf_metrics['roc_auc'],
        'xgb_accuracy': xgb_metrics['accuracy'], 'xgb_roc_auc': xgb_metrics['roc_auc'],
    }
    if run_stacked_ensemble and stacked_metrics:
        run_details.update({
            'stacked_accuracy': stacked_metrics['accuracy'],
            'stacked_roc_auc': stacked_metrics['roc_auc'],
        })
    log_run_results(run_details)

def main():
    start_time_total = time.time()
    print("\n--- Starting Heart Disease Analysis ---")

    dask_client = get_dask_client(cluster_type=DASK_TYPE)
    print(f"Dask client created: {dask_client.dashboard_link}")

    try:
        combined_df = run_data_pipeline(VERBOSE_OUTPUT, SHOW_PLOTS)

        if combined_df is not None:
            lr_metrics, rf_metrics, xgb_metrics, stacked_metrics = run_model_pipeline(
                dask_client, combined_df, VERBOSE_OUTPUT, RUN_STACKED_ENSEMBLE, META_CLASSIFIER
            )
            log_analysis_results(start_time_total, DASK_TYPE, lr_metrics, rf_metrics, xgb_metrics, stacked_metrics, RUN_STACKED_ENSEMBLE)

    finally:
        if dask_client:
            print("Closing Dask client...")
            dask_client.close()
            print("Dask client closed.")

    print(f"\n--- Analysis Complete --- Total time: {time.time() - start_time_total:.2f} seconds.")

if __name__ == "__main__":
    main()
