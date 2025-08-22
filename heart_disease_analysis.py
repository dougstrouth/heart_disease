import warnings
import time
import pandas as pd
import numpy as np
import logging
import dask.dataframe as dd
import mlflow # Added for MLflow integration
import mlflow.sklearn
import mlflow.xgboost

# Define MLflow Tracking and Artifact URIs
# For local tracking, use a file URI. For GCS artifact storage, use gs://
MLFLOW_TRACKING_URI = "file:///Users/dougstrouth/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/heart_disease/mlruns"
MLFLOW_ARTIFACT_URI = "gs://my-heart-disease-data-bucket/mlflow-artifacts"

from config import SHOW_PLOTS, VERBOSE_OUTPUT, DASK_TYPE, RUN_STACKED_ENSEMBLE, META_CLASSIFIER, CV_FOLDS, LR_C_OPTIONS, RF_N_ESTIMATORS_OPTIONS, RF_MAX_DEPTH_OPTIONS, RF_MIN_SAMPLES_SPLIT_OPTIONS, RF_MIN_SAMPLES_LEAF_OPTIONS, XGB_N_ESTIMATORS_OPTIONS, XGB_LEARNING_RATE_OPTIONS
from dask_utils import get_dask_client
from data_utils import load_data, combine_and_clean_data, perform_eda, preprocess_data, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
from model_training import train_evaluate_model, train_stacked_model
from model_interpretation import interpret_model
from preprocessing import get_preprocessor, get_feature_names
from utils.logging_utils import log_run_results
from utils.logger_config import setup_logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib

def run_data_pipeline(verbose_output, show_plots):
    logger = logging.getLogger('heart_disease_analysis')
    logger.info("\n--- Data Pipeline ---")
    start_time_load = time.time()

    if DASK_TYPE == 'local':
        # Load and combine individual local datasets
        df_pratyushpuri = load_data('unified_heart_disease_dataset.csv')
        df_edwankarimsony = load_data('/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/edwankarimsony/heart-disease-data/heart_disease_uci.csv')
        df_johnsmith88 = load_data('/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/johnsmith88/heart-disease-data/heart.csv')

        logger.info(f"Data Loading completed in {time.time() - start_time_load:.2f} seconds.")

        start_time_harmonize = time.time()
        # Rename 'target' column to 'heart_disease' in df_johnsmith88 for harmonization
        if df_johnsmith88 is not None and 'target' in df_johnsmith88.columns:
            df_johnsmith88 = df_johnsmith88.rename(columns={'target': 'heart_disease'})

        # Harmonize and combine the first two datasets
        combined_df_initial = combine_and_clean_data(df_pratyushpuri, df_edwankarimsony, verbose_output=verbose_output)

        # Harmonize the third dataset and combine with the initial combined_df
        if combined_df_initial is not None and df_johnsmith88 is not None:
            combined_df = combine_and_clean_data(combined_df_initial, df_johnsmith88, verbose_output=verbose_output)
        else:
            combined_df = None # Handle cases where initial combination failed or third df is None

        logger.info(f"Data Harmonization and Combination completed in {time.time() - start_time_harmonize:.2f} seconds.")

        # Perform EDA and save combined file locally
        if combined_df is not None:
            perform_eda(combined_df, "Combined Dataset", NUMERICAL_FEATURES, CATEGORICAL_FEATURES, show_plots=show_plots, verbose_output=verbose_output)

            output_csv_path = "combined_heart_disease_dataset.csv"
            if isinstance(combined_df, pd.DataFrame):
                combined_df.to_csv(output_csv_path, index=False)
                logger.info(f"Combined dataset saved to {output_csv_path}")
            else: # Assuming it's a Dask DataFrame
                combined_df.to_csv(output_csv_path, index=False, single_file=True)
                logger.info(f"Combined Dask dataset saved to {output_csv_path}")

    else:
        # Load pre-combined dataset from GCS for Coiled runs
        combined_df = load_data('gs://my-heart-disease-data-bucket/data/combined_heart_disease_dataset.csv')
        logger.info(f"Data Loading completed in {time.time() - start_time_load:.2f} seconds.")

        # Perform EDA (no saving needed as it's already in GCS)
        if combined_df is not None:
            perform_eda(combined_df, "Combined Dataset", NUMERICAL_FEATURES, CATEGORICAL_FEATURES, show_plots=show_plots, verbose_output=verbose_output)

    return combined_df



def run_model_pipeline(dask_client, combined_df, verbose_output, run_stacked_ensemble, meta_classifier):
    logger = logging.getLogger('heart_disease_analysis')
    logger.info("\n--- Model Training, Evaluation, and Interpretation ---")

    X = combined_df.drop(TARGET_COLUMN, axis=1)
    y = combined_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if verbose_output: logger.info("\nData split into training and testing sets.")

    preprocessor = get_preprocessor(CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    # Logistic Regression
    lr_param_grid = {'classifier__C': LR_C_OPTIONS}
    lr_model, _, _, lr_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, X_train_processed, X_test_processed, model_type='logistic_regression', param_grid=lr_param_grid, dask_client=dask_client)

    # Random Forest
    rf_param_grid = {
        'classifier__n_estimators': RF_N_ESTIMATORS_OPTIONS,
        'classifier__max_depth': RF_MAX_DEPTH_OPTIONS,
        'classifier__min_samples_split': RF_MIN_SAMPLES_SPLIT_OPTIONS,
        'classifier__min_samples_leaf': RF_MIN_SAMPLES_LEAF_OPTIONS
    }
    rf_model, _, _, rf_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, X_train_processed, X_test_processed, model_type='random_forest', param_grid=rf_param_grid, dask_client=dask_client)

    # XGBoost
    xgb_param_grid = {
        'classifier__n_estimators': XGB_N_ESTIMATORS_OPTIONS,
        'classifier__learning_rate': XGB_LEARNING_RATE_OPTIONS
    }
    xgb_model, _, _, xgb_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, X_train_processed, X_test_processed, model_type='xgboost', param_grid=xgb_param_grid, dask_client=dask_client)

    stacked_model = None
    stacked_metrics = None
    if run_stacked_ensemble:
        base_models = {'lr': lr_model, 'rf': rf_model, 'xgb': xgb_model}
        stacked_model, _, _, stacked_metrics = train_stacked_model(base_models, X_train_processed, y_train, X_test_processed, y_test, meta_classifier, dask_client)

    logger.info("\n--- Model Interpretability (Random Forest) ---")
    if rf_model is not None:
        interpret_model(rf_model, X_train_processed, feature_names)

    return lr_model, lr_metrics, rf_model, rf_metrics, xgb_model, xgb_metrics, stacked_model, stacked_metrics, X, y, preprocessor

def perform_final_model_evaluation(model, X_original, y_original, fitted_preprocessor, model_name, dask_client, verbose_output):
    logger = logging.getLogger('heart_disease_analysis')
    logger.info(f"\n--- Final Cross-Validation for {model_name} ---")

    # Apply the fitted preprocessor to the original data
    X_processed_full = fitted_preprocessor.transform(X_original)
    y_full = y_original

    # The 'model' passed here is already a Pipeline containing the classifier.
    # We don't need to re-wrap it in another pipeline with a preprocessor.
    # The data is already processed.

    if dask_client:
        logger.info(f"Using Dask for cross-validation: {dask_client.dashboard_link}")
        with joblib.parallel_backend('dask'):
            cv_scores = cross_val_score(model, X_processed_full, y_full, cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_processed_full, y_full, cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1)

    logger.info(f"{model_name} Cross-Validation ROC AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    return np.mean(cv_scores), np.std(cv_scores)

def _extract_metrics_from_report(metrics_dict, prefix):
    extracted = {}
    if 'classification_report' in metrics_dict and isinstance(metrics_dict['classification_report'], dict):
        report = metrics_dict['classification_report']
        # Assuming binary classification with '0' and '1' as class labels
        for cls in ['0', '1']:
            if cls in report:
                extracted[f'{prefix}_precision_class{cls}'] = report[cls]['precision']
                extracted[f'{prefix}_recall_class{cls}'] = report[cls]['recall']
                extracted[f'{prefix}_f1_class{cls}'] = report[cls]['f1-score']
        # Add macro avg and weighted avg for completeness
        if 'macro avg' in report:
            extracted[f'{prefix}_precision_macro_avg'] = report['macro avg']['precision']
            extracted[f'{prefix}_recall_macro_avg'] = report['macro avg']['recall']
            extracted[f'{prefix}_f1_macro_avg'] = report['macro avg']['f1-score']
        if 'weighted avg' in report:
            extracted[f'{prefix}_precision_weighted_avg'] = report['weighted avg']['precision']
            extracted[f'{prefix}_recall_weighted_avg'] = report['weighted avg']['recall']
            extracted[f'{prefix}_f1_weighted_avg'] = report['weighted avg']['f1-score']
    return extracted

def log_analysis_results(start_time, dask_type, lr_metrics, rf_metrics, xgb_metrics, stacked_metrics, run_stacked_ensemble, lr_cv_mean=None, lr_cv_std=None, rf_cv_mean=None, rf_cv_std=None, xgb_cv_mean=None, xgb_cv_std=None, stacked_cv_mean=None, stacked_cv_std=None):
    logger = logging.getLogger('heart_disease_analysis')
    run_details = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_runtime_seconds': time.time() - start_time,
        'dask_type': dask_type,
        'lr_accuracy': lr_metrics['accuracy'],
        'lr_roc_auc': lr_metrics['roc_auc'],
        'lr_best_params': str(lr_metrics['best_params']),
        'lr_train_accuracy': lr_metrics['train_accuracy'],
        'lr_train_roc_auc': lr_metrics['train_roc_auc'],
        'lr_cv_mean_roc_auc': lr_cv_mean,
        'lr_cv_std_roc_auc': lr_cv_std,
    }
    run_details.update(_extract_metrics_from_report(lr_metrics, 'lr'))

    run_details.update({
        'rf_accuracy': rf_metrics['accuracy'],
        'rf_roc_auc': rf_metrics['roc_auc'],
        'rf_best_params': str(rf_metrics['best_params']),
        'rf_train_accuracy': rf_metrics['train_accuracy'],
        'rf_train_roc_auc': rf_metrics['train_roc_auc'],
        'rf_cv_mean_roc_auc': rf_cv_mean,
        'rf_cv_std_roc_auc': rf_cv_std,
    })
    run_details.update(_extract_metrics_from_report(rf_metrics, 'rf'))

    run_details.update({
        'xgb_accuracy': xgb_metrics['accuracy'],
        'xgb_roc_auc': xgb_metrics['roc_auc'],
        'xgb_best_params': str(xgb_metrics['best_params']),
        'xgb_train_accuracy': xgb_metrics['train_accuracy'],
        'xgb_train_roc_auc': xgb_metrics['train_roc_auc'],
        'xgb_cv_mean_roc_auc': xgb_cv_mean,
        'xgb_cv_std_roc_auc': xgb_cv_std,
    })
    run_details.update(_extract_metrics_from_report(xgb_metrics, 'xgb'))

    if run_stacked_ensemble and stacked_metrics:
        run_details.update({
            'stacked_accuracy': stacked_metrics['accuracy'],
            'stacked_roc_auc': stacked_metrics['roc_auc'],
            'stacked_best_params': str(stacked_metrics['best_params'] if 'best_params' in stacked_metrics else None),
            'stacked_cv_mean_roc_auc': stacked_cv_mean,
            'stacked_cv_std_roc_auc': stacked_cv_std,
        })
        run_details.update(_extract_metrics_from_report(stacked_metrics, 'stacked'))
    else:
        run_details['stacked_accuracy'] = None
        run_details['stacked_roc_auc'] = None
        run_details['stacked_best_params'] = None
        run_details['stacked_cv_mean_roc_auc'] = None
        run_details['stacked_cv_std_roc_auc'] = None

    log_run_results(run_details)

def run_analysis():
    start_time_total = time.time()
    logger = setup_logging()
    logger.info("\n--- Starting Heart Disease Analysis ---")

    dask_client = get_dask_client(cluster_type=DASK_TYPE)
    logger.info(f"Dask client created: {dask_client.dashboard_link}")

    try:
        combined_df = run_data_pipeline(VERBOSE_OUTPUT, SHOW_PLOTS)

        if combined_df is not None:
            lr_model, lr_metrics, rf_model, rf_metrics, xgb_model, xgb_metrics, stacked_model, stacked_metrics, X, y, preprocessor = run_model_pipeline(
                dask_client, combined_df, VERBOSE_OUTPUT, RUN_STACKED_ENSEMBLE, META_CLASSIFIER
            )

            # Perform final cross-validation for robustness
            lr_cv_mean, lr_cv_std = perform_final_model_evaluation(lr_model, X, y, preprocessor, "Logistic Regression", dask_client, VERBOSE_OUTPUT)
            rf_cv_mean, rf_cv_std = perform_final_model_evaluation(rf_model, X, y, preprocessor, "Random Forest", dask_client, VERBOSE_OUTPUT)
            xgb_cv_mean, xgb_cv_std = perform_final_model_evaluation(xgb_model, X, y, preprocessor, "XGBoost", dask_client, VERBOSE_OUTPUT)

            stacked_cv_mean, stacked_cv_std = None, None
            if RUN_STACKED_ENSEMBLE and stacked_model is not None:
                stacked_cv_mean, stacked_cv_std = perform_final_model_evaluation(stacked_model, X, y, preprocessor, "Stacked Ensemble", dask_client, VERBOSE_OUTPUT)

            log_analysis_results(start_time_total, DASK_TYPE, lr_metrics, rf_metrics, xgb_metrics, stacked_metrics, RUN_STACKED_ENSEMBLE, lr_cv_mean, lr_cv_std, rf_cv_mean, rf_cv_std, xgb_cv_mean, xgb_cv_std, stacked_cv_mean, stacked_cv_std)

            # MLflow Logging
            mlflow.log_metric("lr_cv_roc_auc", float(lr_cv_mean))
            mlflow.log_metric("rf_cv_roc_auc", float(rf_cv_mean))
            mlflow.log_metric("xgb_cv_roc_auc", float(xgb_cv_mean))
            if RUN_STACKED_ENSEMBLE and stacked_cv_mean is not None:
                mlflow.log_metric("stacked_cv_roc_auc", float(stacked_cv_mean))

            # Create a processed input example for MLflow logging
            processed_input_example = preprocessor.transform(X[:5])

            # Log models
            mlflow.sklearn.log_model(lr_model, name="logistic_regression_model", input_example=processed_input_example)  # type: ignore
            mlflow.sklearn.log_model(rf_model, name="random_forest_model", input_example=processed_input_example)  # type: ignore
            mlflow.xgboost.log_model(xgb_model.named_steps['classifier'], name="xgboost_model", input_example=processed_input_example)  # type: ignore
            if RUN_STACKED_ENSEMBLE and stacked_model is not None:
                mlflow.sklearn.log_model(stacked_model, name="stacked_ensemble_model", input_example=processed_input_example)  # type: ignore

    finally:
        if dask_client:
            logger.info("Closing Dask client...")
            dask_client.close()
            logger.info("Dask client closed.")

    logger.info(f"\n--- Analysis Complete --- Total time: {time.time() - start_time_total:.2f} seconds.")

def main():
    run_analysis() # Call the refactored analysis function


if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Heart Disease Prediction")

    with mlflow.start_run():
        # Log Dask type as a parameter
        mlflow.log_param("dask_type", DASK_TYPE)

        main()
