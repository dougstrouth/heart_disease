import warnings
import time
import pandas as pd
import numpy as np
import logging
import dask.dataframe as dd

from config import SHOW_PLOTS, VERBOSE_OUTPUT, DASK_TYPE, RUN_STACKED_ENSEMBLE, META_CLASSIFIER, CV_FOLDS
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
    combined_df = load_data('gs://my-heart-disease-data-bucket/data/combined_heart_disease_dataset.csv')
    logger.info(f"Data Loading completed in {time.time() - start_time_load:.2f} seconds.")

    start_time_harmonize = time.time()
    # No harmonization needed as data is already combined and cleaned
    if combined_df is not None:
        # Ensure target column is binarized and categorical features are handled if not already
        # This part might need to be adjusted based on how 'combined_heart_disease_dataset.csv' was saved
        # and if it retains Dask DataFrame properties or is a simple CSV.
        # Assuming it's a clean CSV, we might need to re-apply some cleaning steps if not already done.
        # For now, we'll assume the saved CSV is ready for direct use.
        # If it's a Dask DataFrame, we might need to compute it if subsequent steps expect pandas.
        # For simplicity, we'll assume load_data returns a Dask DataFrame if DASK_TYPE is coiled.
        is_dask = isinstance(combined_df, dd.DataFrame)
        if is_dask:
            # If it's a Dask DataFrame, ensure it's computed if subsequent steps expect pandas
            # or if operations like .dropna() need to be triggered.
            # For now, we'll rely on Dask-ML to handle Dask DataFrames.
            pass
        else:
            # If it's a pandas DataFrame, ensure it's cleaned
            initial_rows = len(combined_df)
            combined_df.dropna(subset=[TARGET_COLUMN], inplace=True)
            rows_after_dropna = len(combined_df)
            logger.info(f"Dropped {initial_rows - rows_after_dropna} rows with NaN in '{TARGET_COLUMN}'.")
            logger.info(f"Combined dataset now has {rows_after_dropna} rows.")

            for col in CATEGORICAL_FEATURES:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].astype(str)
                    combined_df[col] = combined_df[col].fillna('missing')
                    if verbose_output: logger.info(f"Converted '{col}' to string and filled NaNs.")

            for col in BINARY_FEATURES:
                if col in combined_df.columns:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                    combined_df[col] = combined_df[col].fillna(0) # Fill NaNs with 0 for binary features
                    if verbose_output: logger.info(f"Filled NaNs in binary feature '{col}' with 0.")

            combined_df[TARGET_COLUMN] = combined_df[TARGET_COLUMN].astype(int)
            combined_df[TARGET_COLUMN] = (combined_df[TARGET_COLUMN] > 0).astype(int)
            if verbose_output: logger.info(f"Binarized '{TARGET_COLUMN}': values > 0 converted to 1.")

    logger.info(f"Data Harmonization and Combination completed in {time.time() - start_time_harmonize:.2f} seconds.")

    if combined_df is not None:
        perform_eda(combined_df, "Combined Dataset", NUMERICAL_FEATURES, CATEGORICAL_FEATURES, show_plots=show_plots, verbose_output=verbose_output)

        # Save the combined DataFrame to a CSV file
        output_csv_path = "combined_heart_disease_dataset.csv"
        if isinstance(combined_df, pd.DataFrame):
            combined_df.to_csv(output_csv_path, index=False)
            logger.info(f"Combined dataset saved to {output_csv_path}")
        else: # Assuming it's a Dask DataFrame
            combined_df.to_csv(output_csv_path, index=False, single_file=True)
            logger.info(f"Combined Dask dataset saved to {output_csv_path}")

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
    lr_model, _, _, lr_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, model_type='logistic_regression', param_grid={'classifier__C': [1.0]}, dask_client=dask_client)

    # Random Forest
    rf_model, _, _, rf_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, model_type='random_forest', param_grid={'classifier__n_estimators': [100]}, dask_client=dask_client)

    # XGBoost
    xgb_model, _, _, xgb_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, model_type='xgboost', param_grid={'classifier__n_estimators': [100]}, dask_client=dask_client)

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

def main():
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

    finally:
        if dask_client:
            logger.info("Closing Dask client...")
            dask_client.close()
            logger.info("Dask client closed.")

    logger.info(f"\n--- Analysis Complete --- Total time: {time.time() - start_time_total:.2f} seconds.")

if __name__ == "__main__":
    main()
