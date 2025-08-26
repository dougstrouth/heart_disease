
import time
import logging
import joblib
import numpy as np
from dask import persist
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split # Import scikit-learn version

from config import (
    DASK_TYPE, CV_FOLDS, LR_C_OPTIONS, RF_N_ESTIMATORS_OPTIONS, RF_MAX_DEPTH_OPTIONS, RF_MIN_SAMPLES_SPLIT_OPTIONS, RF_MIN_SAMPLES_LEAF_OPTIONS, XGB_N_ESTIMATORS_OPTIONS, XGB_LEARNING_RATE_OPTIONS,
    COILED_LR_C_OPTIONS, COILED_RF_N_ESTIMATORS_OPTIONS, COILED_RF_MAX_DEPTH_OPTIONS, COILED_RF_MIN_SAMPLES_SPLIT_OPTIONS, COILED_RF_MIN_SAMPLES_LEAF_OPTIONS,
    COILED_XGB_N_ESTIMATORS_OPTIONS, COILED_XGB_LEARNING_RATE_OPTIONS
)
from data_schema import HeartDiseaseSchema
from model_training import train_evaluate_model
from ensemble_utils import train_stacked_model
from model_interpretation import interpret_model
from preprocessing import get_preprocessor, get_feature_names
from utils.logging_utils import log_run_results

def run_model_pipeline(dask_client, combined_df, verbose_output, run_stacked_ensemble, meta_classifier):
    logger = logging.getLogger('heart_disease_analysis')
    logger.info("\n--- Model Training, Evaluation, and Interpretation ---")

    X = combined_df.drop(HeartDiseaseSchema.TARGET_COLUMN, axis=1)
    y = combined_df[HeartDiseaseSchema.TARGET_COLUMN]

    # Convert to pandas for stratified split
    if hasattr(X, 'compute'):
        X_pd = X.compute()
        y_pd = y.compute()
    else:
        X_pd = X
        y_pd = y

    # Ensure all missing values are np.nan for scikit-learn compatibility
    X_pd = X_pd.mask(X_pd.isna(), np.nan)
    y_pd = y_pd.mask(y_pd.isna(), np.nan)

    # Check if stratification is possible
    if y_pd.value_counts().min() < 2:
        X_train, X_test, y_train, y_test = train_test_split(X_pd, y_pd, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_pd, y_pd, test_size=0.2, random_state=42, stratify=y_pd)
    if verbose_output:
        logger.info("\nData split into training and testing sets.")

    binary_features = ['sex', 'fbs', 'exang', 'smoking', 'diabetes']
    preprocessor = get_preprocessor(HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE(), HeartDiseaseSchema.NUMERICAL_COLUMNS(), binary_features, use_dask_ml=False)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Persist Dask objects to cluster memory
    if hasattr(y_train, 'persist'):
        logger.info("Persisting y_train and y_test to Dask cluster memory...")
        y_train, y_test = persist(y_train, y_test)
        logger.info("y_train and y_test persisted.")


    feature_names = get_feature_names(preprocessor)

    # Determine which parameter options to use based on DASK_TYPE
    if DASK_TYPE in ['coiled', 'cloud']:
        current_lr_c_options = COILED_LR_C_OPTIONS
        current_rf_n_estimators_options = COILED_RF_N_ESTIMATORS_OPTIONS
        current_rf_max_depth_options = COILED_RF_MAX_DEPTH_OPTIONS
        current_rf_min_samples_split_options = COILED_RF_MIN_SAMPLES_SPLIT_OPTIONS
        current_rf_min_samples_leaf_options = COILED_RF_MIN_SAMPLES_LEAF_OPTIONS
        current_xgb_n_estimators_options = COILED_XGB_N_ESTIMATORS_OPTIONS
        current_xgb_learning_rate_options = COILED_XGB_LEARNING_RATE_OPTIONS
    else:
        current_lr_c_options = LR_C_OPTIONS
        current_rf_n_estimators_options = RF_N_ESTIMATORS_OPTIONS
        current_rf_max_depth_options = RF_MAX_DEPTH_OPTIONS
        current_rf_min_samples_split_options = RF_MIN_SAMPLES_SPLIT_OPTIONS
        current_rf_min_samples_leaf_options = RF_MIN_SAMPLES_LEAF_OPTIONS
        current_xgb_n_estimators_options = XGB_N_ESTIMATORS_OPTIONS
        current_xgb_learning_rate_options = XGB_LEARNING_RATE_OPTIONS

    # Logistic Regression
    lr_param_grid = {'classifier__C': current_lr_c_options}
    lr_model, _, _, lr_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, X_train_processed, X_test_processed, model_type='logistic_regression', param_grid=lr_param_grid, dask_client=dask_client)

    # Random Forest
    rf_param_grid = {
        'classifier__n_estimators': current_rf_n_estimators_options,
        'classifier__max_depth': current_rf_max_depth_options,
        'classifier__min_samples_split': current_rf_min_samples_split_options,
        'classifier__min_samples_leaf': current_rf_min_samples_leaf_options
    }
    rf_model, _, _, rf_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, X_train_processed, X_test_processed, model_type='random_forest', param_grid=rf_param_grid, dask_client=dask_client)

    # XGBoost
    xgb_param_grid = {
        'classifier__n_estimators': current_xgb_n_estimators_options,
        'classifier__learning_rate': current_xgb_learning_rate_options
    }
    xgb_model, _, _, xgb_metrics = train_evaluate_model(X_train_processed, y_train, X_test_processed, y_test, X_train_processed, X_test_processed, model_type='xgboost', param_grid=xgb_param_grid, dask_client=dask_client)

    stacked_model = None
    stacked_metrics = None
    if run_stacked_ensemble:
        base_models = {'lr': lr_model, 'rf': rf_model, 'xgb': xgb_model}
        stacked_model, _, _, stacked_metrics, stacked_input_example = train_stacked_model(base_models, X_train_processed, y_train, X_test_processed, y_test, meta_classifier, dask_client)

    logger.info("\n--- Model Interpretability (Random Forest) ---")
    if rf_model is not None:
        interpret_model(rf_model, X_train_processed, feature_names)

    return lr_model, lr_metrics, rf_model, rf_metrics, xgb_model, xgb_metrics, stacked_model, stacked_metrics, X, y, preprocessor, stacked_input_example


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
        # Assuming binary classification with '0', '1' as class labels
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
    # logger = logging.getLogger('heart_disease_analysis') # Removed unused logger assignment
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

    log_run_results(run_details)
