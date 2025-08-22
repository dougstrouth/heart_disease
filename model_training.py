import time
import joblib
import os
from typing import Optional, Any
import logging

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost # Import for XGBoost models

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

from dask.distributed import Client

# Import configuration options
from config import LR_C_OPTIONS, RF_N_ESTIMATORS_OPTIONS, RF_MAX_DEPTH_OPTIONS, RF_MIN_SAMPLES_SPLIT_OPTIONS, RF_MIN_SAMPLES_LEAF_OPTIONS, XGB_N_ESTIMATORS_OPTIONS, XGB_LEARNING_RATE_OPTIONS, RF_RANDOM_SEARCH_N_ITER, LR_RANDOM_SEARCH_N_ITER, XGB_RANDOM_SEARCH_N_ITER, CV_FOLDS, RUN_STACKED_ENSEMBLE, META_CLASSIFIER

# Import stacking utility
from ensemble_utils import train_stacked_model

def train_evaluate_model(X_train, y_train, X_test, y_test, model_type='logistic_regression', param_grid=None, dask_client: Optional[Client] = None):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Trains and evaluates a specified machine learning model.
    If param_grid is provided, performs GridSearchCV for hyperparameter tuning.
    """
    if X_train is None or y_train is None:
        logger.error("Cannot train/evaluate: Data is None.")
        return None, None, None, None

    # MLflow will handle model caching, so we can remove the joblib caching logic here
    # cache_dir = "cache/models"
    # os.makedirs(cache_dir, exist_ok=True)
    # model_cache_path = os.path.join(cache_dir, f"{model_type}_model.joblib")

    best_model: Any
    # if os.path.exists(model_cache_path) and not param_grid:
    #     logger.info(f"\nLoading {model_type} model from cache...")
    #     best_model = joblib.load(model_cache_path)
    #     logger.info("Model loaded from cache.")

    #     y_pred = best_model.predict(X_test)
    #     y_proba = best_model.predict_proba(X_test)[:, 1]

    #     accuracy = accuracy_score(y_test, y_pred)
    #     precision = precision_score(y_test, y_pred)
    #     recall = recall_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)
    #     roc_auc = roc_auc_score(y_test, y_proba)
    #     conf_matrix = confusion_matrix(y_test, y_pred)
    #     class_report_dict = classification_report(y_test, y_pred, output_dict=True)

    #     logger.info(f"\n--- {model_type} Model Evaluation (Loaded from Cache) ---")
    #     logger.info(f"Accuracy: {accuracy:.4f}")
    #     logger.info(f"Precision: {precision:.4f}")
    #     logger.info(f"Recall: {recall:.4f}")
    #     logger.info(f"F1-Score: {f1:.4f}")
    #     logger.info(f"ROC AUC: {roc_auc:.4f}")

    #     logger.info("\nConfusion Matrix:")
    #     logger.info(conf_matrix)

    #     logger.info("\nClassification Report:")
    #     logger.info(classification_report(y_test, y_pred)) # Print readable format

    #     metrics = {
    #         'accuracy': accuracy,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1_score': f1,
    #         'roc_auc': roc_auc,
    #         'confusion_matrix': conf_matrix,
    #         'classification_report': class_report_dict,
    #         'best_params': None, # No grid search performed
    #         'best_cv_score': None
    #     }
    #     return best_model, y_pred, y_proba, metrics

    # else:
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

    if param_grid:
        logger.info(f"\nPerforming GridSearchCV for {model_name}...")
        if model_type == 'random_forest':
            search = RandomizedSearchCV(model_pipeline, param_grid, cv=CV_FOLDS, scoring='roc_auc', n_iter=RF_RANDOM_SEARCH_N_ITER, n_jobs=-1, verbose=1)
        elif model_type == 'logistic_regression':
            search = RandomizedSearchCV(model_pipeline, param_grid, cv=CV_FOLDS, scoring='roc_auc', n_iter=LR_RANDOM_SEARCH_N_ITER, n_jobs=-1, verbose=1)
        elif model_type == 'xgboost':
            search = RandomizedSearchCV(model_pipeline, param_grid, cv=CV_FOLDS, scoring='roc_auc', n_iter=XGB_RANDOM_SEARCH_N_ITER, n_jobs=-1, verbose=1)
        else:
            search = GridSearchCV(model_pipeline, param_grid, cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1, verbose=1)

        if dask_client:
            logger.info(f"Using Dask for parallel processing: {dask_client.dashboard_link}")
            with joblib.parallel_backend('dask'):
                search.fit(X_train, y_train)
        else:
            search.fit(X_train, y_train)

        best_model = search.best_estimator_
        logger.info(f"GridSearchCV complete for {model_name}.")
        logger.info(f"Best parameters for {model_name}: {search.best_params_}")
        logger.info(f"Best ROC AUC score for {model_name}: {search.best_score_:.4f}")

        # MLflow: Log best parameters and best CV score
        mlflow.log_params(search.best_params_)
        mlflow.log_metric(f"{model_type}_best_cv_roc_auc", search.best_score_)

    else:
        logger.info(f"\nTraining {model_name} model (without tuning)...")
        model_pipeline.fit(X_train, y_train)
        best_model = model_pipeline
        logger.info("Training complete.")

    # MLflow: Log the trained model
    if model_type == 'logistic_regression':
        mlflow.sklearn.log_model(best_model, "logistic_regression_model")
    elif model_type == 'random_forest':
        mlflow.sklearn.log_model(best_model, "random_forest_model")
    elif model_type == 'xgboost':
        mlflow.xgboost.log_model(best_model, "xgboost_model")

    # joblib.dump(best_model, model_cache_path)
    # logger.info(f"Model cached to {model_cache_path}")

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
    mlflow.log_metric(f"{model_type}_accuracy", accuracy)
    mlflow.log_metric(f"{model_type}_precision", precision)
    mlflow.log_metric(f"{model_type}_recall", recall)
    mlflow.log_metric(f"{model_type}_f1_score", f1)
    mlflow.log_metric(f"{model_type}_roc_auc", roc_auc)
    mlflow.log_metric(f"{model_type}_train_accuracy", train_accuracy)
    mlflow.log_metric(f"{model_type}_train_roc_auc", train_roc_auc)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report_dict,
        'best_params': search.best_params_ if param_grid else None,
        'best_cv_score': search.best_score_ if param_grid else None,
        'train_accuracy': train_accuracy,
        'train_roc_auc': train_roc_auc
    }

    return best_model, y_pred, y_proba, metrics
