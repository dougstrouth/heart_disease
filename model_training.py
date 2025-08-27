import joblib
from typing import Optional, Any
import logging

import mlflow
import mlflow.sklearn
import mlflow.xgboost # Import for XGBoost models

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

from dask.distributed import Client
import dask.dataframe as dd # Import dask.dataframe

# Import configuration options
from config import (
    DASK_TYPE, RF_RANDOM_SEARCH_N_ITER, LR_RANDOM_SEARCH_N_ITER, XGB_RANDOM_SEARCH_N_ITER, CV_FOLDS,
    LR_C_OPTIONS, RF_N_ESTIMATORS_OPTIONS, RF_MAX_DEPTH_OPTIONS, RF_MIN_SAMPLES_SPLIT_OPTIONS, RF_MIN_SAMPLES_LEAF_OPTIONS,
    XGB_N_ESTIMATORS_OPTIONS, XGB_LEARNING_RATE_OPTIONS,
    COILED_LR_C_OPTIONS, COILED_RF_N_ESTIMATORS_OPTIONS, COILED_RF_MAX_DEPTH_OPTIONS, COILED_RF_MIN_SAMPLES_SPLIT_OPTIONS, COILED_RF_MIN_SAMPLES_LEAF_OPTIONS,
    COILED_XGB_N_ESTIMATORS_OPTIONS, COILED_XGB_LEARNING_RATE_OPTIONS,
    COILED_RF_RANDOM_SEARCH_N_ITER, COILED_LR_RANDOM_SEARCH_N_ITER, COILED_XGB_RANDOM_SEARCH_N_ITER
)


# Import stacking utility

def train_evaluate_model(X_train, y_train, X_test, y_test, X_train_processed, X_test_processed, model_type='logistic_regression', param_grid=None, dask_client: Optional[Client] = None, best_params: Optional[dict] = None):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Trains and evaluates a specified machine learning model.
    If param_grid is provided, performs GridSearchCV for hyperparameter tuning.
    If best_params are provided, uses those parameters directly.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot train/evaluate: Data is None.")
        return None, None, None, None

    best_model: Any

    if model_type == 'logistic_regression':
        classifier = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        model_name = "Logistic Regression"
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
        model_name = "Random Forest"
    elif model_type == 'xgboost':
        classifier = XGBClassifier(eval_metric='logloss', random_state=42)
        model_name = "XGBoost"
    else:
        logger.error(f"Error: Unknown model type '{model_type}'.")
        return None, None, None, None

    model_pipeline = Pipeline(steps=[('classifier', classifier)])

    # If best_params are provided, use them directly
    if best_params:
        logger.info(f"\nTraining {model_name} model with provided best parameters...")
        # Map Optuna parameter names to sklearn pipeline parameter names
        # This mapping needs to be robust for different model types
        model_specific_best_params = {}
        if model_type == 'logistic_regression':
            if 'lr_C' in best_params: # Assuming Optuna param name is 'lr_C'
                model_specific_best_params['classifier__C'] = best_params['lr_C']
        elif model_type == 'random_forest':
            if 'rf_n_estimators' in best_params:
                model_specific_best_params['classifier__n_estimators'] = best_params['rf_n_estimators']
            if 'rf_max_depth' in best_params:
                model_specific_best_params['classifier__max_depth'] = best_params['rf_max_depth']
            if 'rf_min_samples_split' in best_params:
                model_specific_best_params['classifier__min_samples_split'] = best_params['rf_min_samples_split']
            if 'rf_min_samples_leaf' in best_params:
                model_specific_best_params['classifier__min_samples_leaf'] = best_params['rf_min_samples_leaf']
        elif model_type == 'xgboost':
            if 'xgb_n_estimators' in best_params:
                model_specific_best_params['classifier__n_estimators'] = best_params['xgb_n_estimators']
            if 'xgb_learning_rate' in best_params:
                model_specific_best_params['classifier__learning_rate'] = best_params['xgb_learning_rate']

        # Apply best parameters to the classifier
        classifier.set_params(**model_specific_best_params)
        model_pipeline = Pipeline(steps=[('classifier', classifier)])
        model_pipeline.fit(X_train, y_train)
        best_model = model_pipeline
        logger.info("Training complete with best parameters.")
        # Set search_best_params and search_best_score for logging consistency
        search_best_params = model_specific_best_params
        search_best_score = None # No CV score from a direct fit

    elif param_grid: # Original logic for RandomizedSearchCV
        logger.info(f"\nPerforming GridSearchCV for {model_name}...")

        # Determine which parameter options and n_iter to use based on DASK_TYPE
        if DASK_TYPE in ['coiled', 'cloud']:
            lr_c_options = COILED_LR_C_OPTIONS
            rf_n_estimators_options = COILED_RF_N_ESTIMATORS_OPTIONS
            rf_max_depth_options = COILED_RF_MAX_DEPTH_OPTIONS
            rf_min_samples_split_options = COILED_RF_MIN_SAMPLES_SPLIT_OPTIONS
            rf_min_samples_leaf_options = COILED_RF_MIN_SAMPLES_LEAF_OPTIONS
            xgb_n_estimators_options = COILED_XGB_N_ESTIMATORS_OPTIONS
            xgb_learning_rate_options = COILED_XGB_LEARNING_RATE_OPTIONS
            rf_n_iter = COILED_RF_RANDOM_SEARCH_N_ITER
            lr_n_iter = COILED_LR_RANDOM_SEARCH_N_ITER
            xgb_n_iter = COILED_XGB_RANDOM_SEARCH_N_ITER
        else:
            # Use default options from config for local/pandas runs
            lr_c_options = param_grid.get('classifier__C', LR_C_OPTIONS) # Assuming param_grid might contain these
            rf_n_estimators_options = param_grid.get('classifier__n_estimators', RF_N_ESTIMATORS_OPTIONS)
            rf_max_depth_options = param_grid.get('classifier__max_depth', RF_MAX_DEPTH_OPTIONS)
            rf_min_samples_split_options = param_grid.get('classifier__min_samples_split', RF_MIN_SAMPLES_SPLIT_OPTIONS)
            rf_min_samples_leaf_options = param_grid.get('classifier__min_samples_leaf', RF_MIN_SAMPLES_LEAF_OPTIONS)
            xgb_n_estimators_options = param_grid.get('classifier__n_estimators', XGB_N_ESTIMATORS_OPTIONS)
            xgb_learning_rate_options = param_grid.get('classifier__learning_rate', XGB_LEARNING_RATE_OPTIONS)
            rf_n_iter = RF_RANDOM_SEARCH_N_ITER
            lr_n_iter = LR_RANDOM_SEARCH_N_ITER
            xgb_n_iter = XGB_RANDOM_SEARCH_N_ITER

        # If a param_grid is provided, use it directly. Otherwise, reconstruct based on DASK_TYPE.
        if param_grid:
            current_param_grid = param_grid
            # When param_grid is provided, n_iter should typically be 1 for RandomizedSearchCV
            # as Optuna is handling the search.
            n_iter_to_use = 1
        else:
            # Reconstruct param_grid based on selected options (from config)
            if model_type == 'logistic_regression':
                current_param_grid = {'classifier__C': lr_c_options}
                n_iter_to_use = lr_n_iter
            elif model_type == 'random_forest':
                current_param_grid = {
                    'classifier__n_estimators': rf_n_estimators_options,
                    'classifier__max_depth': rf_max_depth_options,
                    'classifier__min_samples_split': rf_min_samples_split_options,
                    'classifier__min_samples_leaf': rf_min_samples_leaf_options
                }
                n_iter_to_use = rf_n_iter
            elif model_type == 'xgboost':
                current_param_grid = {
                    'classifier__n_estimators': xgb_n_estimators_options,
                    'classifier__learning_rate': xgb_learning_rate_options
                }
                n_iter_to_use = xgb_n_iter
            else:
                # Fallback for other model types, use original param_grid if provided
                current_param_grid = param_grid # This param_grid would be None here if not provided initially
                n_iter_to_use = 20 # Default n_iter if not specified for other types

        # Ensure n_iter does not exceed the total number of combinations
        # For RandomizedSearchCV, n_iter should be <= total combinations
        # For GridSearchCV, n_iter is not applicable, it runs all combinations
        total_combinations = 1
        for values in current_param_grid.values():
            total_combinations *= len(values)
        n_iter_to_use = min(n_iter_to_use, total_combinations)


        if model_type == 'random_forest' or model_type == 'logistic_regression' or model_type == 'xgboost':
            search = RandomizedSearchCV(model_pipeline, current_param_grid, cv=CV_FOLDS, scoring='roc_auc', n_iter=n_iter_to_use, n_jobs=-1 if model_type != 'logistic_regression' else 1, verbose=1)
        else:
            search = GridSearchCV(model_pipeline, current_param_grid, cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1, verbose=1)

        if dask_client:
            logger.info(f"Using Dask for parallel processing: {dask_client.dashboard_link}")
            with joblib.parallel_backend('dask'):
                search.fit(X_train, y_train)
        else:
            search.fit(X_train, y_train)

        best_model = search.best_estimator_
        search_best_params = search.best_params_
        search_best_score = search.best_score_
        logger.info(f"GridSearchCV complete for {model_name}.")
        logger.info(f"Best parameters for {model_name}: {search_best_params}")
        logger.info(f"Best ROC AUC score for {model_name}: {search_best_score:.4f}")

    else: # Fallback if no best_params and no param_grid
        logger.info(f"\nTraining {model_name} model (without tuning)...\n")
        model_pipeline = Pipeline(steps=[('classifier', classifier)])
        model_pipeline.fit(X_train, y_train)
        best_model = model_pipeline
        logger.info("Training complete.")
        search_best_params = None
        search_best_score = None

    # MLflow: Log best parameters and best CV score (use search_best_params and search_best_score)
    if search_best_params:
        for param_name, param_value in search_best_params.items():
            mlflow.log_param(f"{model_type}_{param_name}", param_value)
    if search_best_score is not None:
        mlflow.log_metric(f"{model_type}_best_cv_roc_auc", search_best_score)

    # MLflow: Log the trained model
    # Ensure input_example is a Dask DataFrame head if X_train_processed is Dask
    if isinstance(X_train_processed, dd.DataFrame):
        input_example_for_mlflow = X_train_processed.head(5)
    else:
        input_example_for_mlflow = X_train_processed[:5]

    if model_type == 'logistic_regression':
        mlflow.sklearn.log_model(best_model, name="logistic_regression_model", input_example=input_example_for_mlflow)  # type: ignore
    elif model_type == 'random_forest':
        mlflow.sklearn.log_model(best_model, name="random_forest_model", input_example=input_example_for_mlflow)  # type: ignore
    elif model_type == 'xgboost':
        mlflow.xgboost.log_model(best_model.named_steps['classifier'], name="xgboost_model", input_example=input_example_for_mlflow)  # type: ignore

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)

    y_pred_train = best_model.predict(X_train)
    y_proba_train = best_model.predict_proba(X_train)[:, 1]

    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_roc_auc = roc_auc_score(y_train, y_proba_train)

    logger.info(f"\n--- {model_name} Model Evaluation (Best Estimator) ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    # MLflow: Log final evaluation metrics
    mlflow.log_metric(f"{model_type}_accuracy", float(accuracy))
    mlflow.log_metric(f"{model_type}_precision", float(precision))
    mlflow.log_metric(f"{model_type}_recall", float(recall))
    mlflow.log_metric(f"{model_type}_f1_score", float(f1))
    mlflow.log_metric(f"{model_type}_roc_auc", float(roc_auc))
    mlflow.log_metric(f"{model_type}_train_accuracy", float(train_accuracy))
    mlflow.log_metric(f"{model_type}_train_roc_auc", float(train_roc_auc))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report_dict,
        'best_params': search_best_params,
        'best_cv_score': search_best_score,
        'train_accuracy': train_accuracy,
        'train_roc_auc': train_roc_auc
    }

    return best_model, y_pred, y_proba, metrics
