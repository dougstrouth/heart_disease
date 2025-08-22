import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from dask.distributed import Client
from typing import Optional
import logging

def train_stacked_model(base_models: dict, X_train, y_train, X_test, y_test, meta_classifier, dask_client: Optional[Client] = None, n_splits_skf=5):
    logger = logging.getLogger('heart_disease_analysis')
    """
    Trains a stacked ensemble model.

    Args:
        base_models (dict): A dictionary of trained base models (e.g., {'lr': lr_model, 'rf': rf_model}).
        X_train, y_train: Training data for base models.
        X_test, y_test: Testing data for base models.
        meta_classifier: The meta-model (e.g., LogisticRegression()).
        dask_client (Client): The Dask client for distributed computation. Note: This client is passed but the
                              sklearn models themselves are not Dask-aware. For distributed stacking,
                              Dask-ML compatible models and cross-validation strategies would be needed.
        n_splits_skf (int): Number of splits for StratifiedKFold.

    Returns:
        tuple: (stacked_model, y_pred, y_proba, metrics)
    """
    logger.info("\n--- Training Stacked Ensemble Model ---")

    # Generate out-of-fold predictions for base models on training data
    # This prevents data leakage from base models to the meta-model
    skf = StratifiedKFold(n_splits=n_splits_skf, shuffle=True, random_state=42)
    oof_predictions = pd.DataFrame()

    for model_name, model in base_models.items():
        oof_preds = np.zeros(X_train.shape[0])
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_X_train, fold_y_train = X_train[train_idx], y_train.iloc[train_idx]
            fold_X_val, fold_y_val = X_train[val_idx], y_train.iloc[val_idx]

            # Fit base model on fold training data
            model.fit(fold_X_train, fold_y_train)
            # Store predictions on fold validation data
            oof_preds[val_idx] = model.predict_proba(fold_X_val)[:, 1]
        oof_predictions[model_name] = oof_preds

    # Generate predictions for base models on test data
    test_predictions = pd.DataFrame()
    for model_name, model in base_models.items():
        test_predictions[model_name] = model.predict_proba(X_test)[:, 1]

    # Train meta-classifier
    logger.info("Training meta-classifier...")
    meta_classifier.fit(oof_predictions, y_train)

    # Make predictions with stacked model
    y_pred_stacked = meta_classifier.predict(test_predictions)
    y_proba_stacked = meta_classifier.predict_proba(test_predictions)[:, 1]

    # Evaluate stacked model
    accuracy = accuracy_score(y_test, y_pred_stacked)
    precision = precision_score(y_test, y_pred_stacked)
    recall = recall_score(y_test, y_pred_stacked)
    f1 = f1_score(y_test, y_pred_stacked)
    roc_auc = roc_auc_score(y_test, y_proba_stacked)
    conf_matrix = confusion_matrix(y_test, y_pred_stacked)
    class_report_dict = classification_report(y_test, y_pred_stacked, output_dict=True)

    logger.info(f"\n--- Stacked Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    logger.info("\nConfusion Matrix:")
    logger.info(conf_matrix)

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred_stacked)) # Print readable format

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report_dict,
    }

    return meta_classifier, y_pred_stacked, y_proba_stacked, metrics
