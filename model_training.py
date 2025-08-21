import time
import joblib
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV, RandomizedSearchCV as DaskRandomizedSearchCV

# Import configuration options
from config import LR_C_OPTIONS, RF_N_ESTIMATORS_OPTIONS, RF_MAX_DEPTH_OPTIONS, RF_MIN_SAMPLES_SPLIT_OPTIONS, RF_MIN_SAMPLES_LEAF_OPTIONS, XGB_N_ESTIMATORS_OPTIONS, XGB_LEARNING_RATE_OPTIONS, RF_RANDOM_SEARCH_N_ITER, RUN_STACKED_ENSEMBLE, META_CLASSIFIER

# Import stacking utility
from ensemble_utils import train_stacked_model

def train_evaluate_model(X_train, y_train, X_test, y_test, preprocessor, model_type='logistic_regression', param_grid=None, dask_client: Client = None):
    """
    Trains and evaluates a specified machine learning model.
    If param_grid is provided, performs GridSearchCV for hyperparameter tuning.
    """
    if X_train is None or y_train is None or preprocessor is None:
        print("Cannot train/evaluate: Data or preprocessor is None.")
        return None, None, None, None

    # Define cache path for the model
    cache_dir = "cache/models"
    os.makedirs(cache_dir, exist_ok=True)
    model_cache_path = os.path.join(cache_dir, f"{model_type}_model.joblib")

    # Check if cached model exists and is valid (simple check for now)
    if os.path.exists(model_cache_path) and not param_grid: # Only load from cache if not doing grid search
        print(f"\nLoading {model_type} model from cache...")
        best_model = joblib.load(model_cache_path)
        print("Model loaded from cache.")

        # Re-evaluate metrics with loaded model (important for consistent logging)
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"\n--- {model_type} Model Evaluation (Loaded from Cache) ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(conf_matrix)

        print("\nClassification Report:")
        print(class_report)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'best_params': None, # No grid search performed
            'best_cv_score': None
        }
        return best_model, y_pred, y_proba, metrics

    if model_type == 'logistic_regression':
        classifier = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        model_name = "Logistic Regression"
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
        model_name = "Random Forest"
    elif model_type == 'xgboost': # Add XGBoost
        classifier = XGBClassifier(eval_metric='logloss', random_state=42)
        model_name = "XGBoost"
    else:
        print(f"Error: Unknown model type '{model_type}'.")
        return None, None, None, None

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    if param_grid:
        print(f"\nPerforming GridSearchCV for {model_name}...")
        if dask_client:
            print(f"Using Dask-ML GridSearchCV with client: {dask_client.dashboard_link}")
            if model_type == 'random_forest':
                grid_search = DaskRandomizedSearchCV(model_pipeline, param_grid, cv=10, scoring='roc_auc', n_iter=RF_RANDOM_SEARCH_N_ITER)
                print(f"Using Dask-ML RandomizedSearchCV for Random Forest with n_iter={RF_RANDOM_SEARCH_N_ITER}")
            else:
                grid_search = DaskGridSearchCV(model_pipeline, param_grid, cv=10, scoring='roc_auc')
        else:
            grid_search = GridSearchCV(model_pipeline, param_grid, cv=10, scoring='roc_auc', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"GridSearchCV complete for {model_name}.")
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best ROC AUC score for {model_name}: {grid_search.best_score_:.4f}")
    else:
        print(f"\nTraining {model_name} model (without tuning)...")
        model_pipeline.fit(X_train, y_train)
        best_model = model_pipeline
        print("Training complete.")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\n--- {model_name} Model Evaluation (Best Estimator) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(class_report)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'best_params': grid_search.best_params_ if param_grid else None,
        'best_cv_score': grid_search.best_score_ if param_grid else None
    }

    # Cache the trained model
    joblib.dump(best_model, model_cache_path)
    print(f"Model cached to {model_cache_path}")

    return best_model, y_pred, y_proba, metrics
