import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from model_training import train_evaluate_model
from config import LR_C_OPTIONS, RF_N_ESTIMATORS_OPTIONS, RF_MAX_DEPTH_OPTIONS, RF_MIN_SAMPLES_SPLIT_OPTIONS, RF_MIN_SAMPLES_LEAF_OPTIONS, XGB_N_ESTIMATORS_OPTIONS, XGB_LEARNING_RATE_OPTIONS, LR_RANDOM_SEARCH_N_ITER, RF_RANDOM_SEARCH_N_ITER, XGB_RANDOM_SEARCH_N_ITER, CV_FOLDS

# Dummy data for testing
X_train_processed = pd.DataFrame(np.random.rand(100, 10))
y_train = pd.Series(np.random.randint(0, 2, 100))
X_test_processed = pd.DataFrame(np.random.rand(50, 10))
y_test = pd.Series(np.random.randint(0, 2, 50))

@pytest.fixture(autouse=True)
def mock_mlflow():
    with (
        patch('mlflow.log_param') as mock_log_param,
        patch('mlflow.log_metric') as mock_log_metric,
        patch('mlflow.sklearn.log_model') as mock_log_sklearn_model,
        patch('mlflow.xgboost.log_model') as mock_log_xgboost_model
    ):
        yield mock_log_param, mock_log_metric, mock_log_sklearn_model, mock_log_xgboost_model

@patch('model_training.RandomizedSearchCV')
def test_train_evaluate_model_lr_n_jobs(mock_randomized_search_cv, mock_mlflow):
    # Configure the mock to return a mock estimator
    mock_instance = MagicMock()
    mock_instance.best_estimator_ = MagicMock()
    mock_instance.best_estimator_.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
    mock_instance.best_estimator_.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
    mock_instance.best_score_ = 0.85
    mock_instance.best_params_ = {'classifier__C': 1.0}
    mock_randomized_search_cv.return_value = mock_instance

    train_evaluate_model(
        X_train_processed, y_train, X_test_processed, y_test,
        X_train_processed, X_test_processed, model_type='logistic_regression',
        param_grid={'classifier__C': LR_C_OPTIONS}
    )

    # Assert that RandomizedSearchCV was called with n_jobs=1 for Logistic Regression
    assert isinstance(mock_randomized_search_cv.call_args[0][0], Pipeline)
    assert isinstance(mock_randomized_search_cv.call_args[0][0].named_steps['classifier'], LogisticRegression)
    assert mock_randomized_search_cv.call_args[0][1] == {'classifier__C': LR_C_OPTIONS}
    assert mock_randomized_search_cv.call_args[1]['cv'] == CV_FOLDS
    assert mock_randomized_search_cv.call_args[1]['scoring'] == 'roc_auc'
    assert mock_randomized_search_cv.call_args[1]['n_iter'] == LR_RANDOM_SEARCH_N_ITER
    assert mock_randomized_search_cv.call_args[1]['n_jobs'] == 1
    assert mock_randomized_search_cv.call_args[1]['verbose'] == 1

@patch('model_training.RandomizedSearchCV')
def test_train_evaluate_model_rf_param_grid(mock_randomized_search_cv, mock_mlflow):
    # Configure the mock to return a mock estimator
    mock_instance = MagicMock()
    mock_instance.best_estimator_ = MagicMock()
    mock_instance.best_estimator_.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
    mock_instance.best_estimator_.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
    mock_instance.best_score_ = 0.90
    mock_instance.best_params_ = {'classifier__n_estimators': 100}
    mock_randomized_search_cv.return_value = mock_instance

    train_evaluate_model(
        X_train_processed, y_train, X_test_processed, y_test,
        X_train_processed, X_test_processed, model_type='random_forest',
        param_grid={
            'classifier__n_estimators': RF_N_ESTIMATORS_OPTIONS,
            'classifier__max_depth': RF_MAX_DEPTH_OPTIONS,
            'classifier__min_samples_split': RF_MIN_SAMPLES_SPLIT_OPTIONS,
            'classifier__min_samples_leaf': RF_MIN_SAMPLES_LEAF_OPTIONS
        }
    )

    # Assert that RandomizedSearchCV was called with the correct param_grid for Random Forest
    assert isinstance(mock_randomized_search_cv.call_args[0][0], Pipeline)
    assert isinstance(mock_randomized_search_cv.call_args[0][0].named_steps['classifier'], RandomForestClassifier)
    assert mock_randomized_search_cv.call_args[0][1] == {
        'classifier__n_estimators': RF_N_ESTIMATORS_OPTIONS,
        'classifier__max_depth': RF_MAX_DEPTH_OPTIONS,
        'classifier__min_samples_split': RF_MIN_SAMPLES_SPLIT_OPTIONS,
        'classifier__min_samples_leaf': RF_MIN_SAMPLES_LEAF_OPTIONS
    }
    assert mock_randomized_search_cv.call_args[1]['cv'] == CV_FOLDS
    assert mock_randomized_search_cv.call_args[1]['scoring'] == 'roc_auc'
    assert mock_randomized_search_cv.call_args[1]['n_iter'] == RF_RANDOM_SEARCH_N_ITER
    assert mock_randomized_search_cv.call_args[1]['n_jobs'] == -1
    assert mock_randomized_search_cv.call_args[1]['verbose'] == 1

@patch('model_training.RandomizedSearchCV')
def test_train_evaluate_model_xgb_param_grid(mock_randomized_search_cv, mock_mlflow):
    # Configure the mock to return a mock estimator
    mock_instance = MagicMock()
    mock_instance.best_estimator_ = MagicMock()
    mock_instance.best_estimator_.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
    mock_instance.best_estimator_.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
    mock_instance.best_score_ = 0.88
    mock_instance.best_params_ = {'classifier__n_estimators': 100}
    mock_randomized_search_cv.return_value = mock_instance

    train_evaluate_model(
        X_train_processed, y_train, X_test_processed, y_test,
        X_train_processed, X_test_processed, model_type='xgboost',
        param_grid={
            'classifier__n_estimators': XGB_N_ESTIMATORS_OPTIONS,
            'classifier__learning_rate': XGB_LEARNING_RATE_OPTIONS
        }
    )

    # Assert that RandomizedSearchCV was called with the correct param_grid for XGBoost
    assert isinstance(mock_randomized_search_cv.call_args[0][0], Pipeline)
    assert isinstance(mock_randomized_search_cv.call_args[0][0].named_steps['classifier'], XGBClassifier)
    assert mock_randomized_search_cv.call_args[0][1] == {
        'classifier__n_estimators': XGB_N_ESTIMATORS_OPTIONS,
        'classifier__learning_rate': XGB_LEARNING_RATE_OPTIONS
    }
    assert mock_randomized_search_cv.call_args[1]['cv'] == CV_FOLDS
    assert mock_randomized_search_cv.call_args[1]['scoring'] == 'roc_auc'
    assert mock_randomized_search_cv.call_args[1]['n_iter'] == XGB_RANDOM_SEARCH_N_ITER
    assert mock_randomized_search_cv.call_args[1]['n_jobs'] == -1
    assert mock_randomized_search_cv.call_args[1]['verbose'] == 1

def test_train_evaluate_model_mlflow_logging(mock_mlflow):
    mock_log_param, mock_log_metric, mock_log_sklearn_model, mock_log_xgboost_model = mock_mlflow

    # Mock RandomizedSearchCV to return a simple estimator
    with patch('model_training.RandomizedSearchCV') as mock_randomized_search_cv:
        mock_instance = MagicMock()
        mock_instance.best_estimator_ = MagicMock()
        mock_instance.best_estimator_.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
        mock_instance.best_estimator_.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
        mock_instance.best_score_ = 0.85
        mock_instance.best_params_ = {'classifier__C': 1.0}
        mock_randomized_search_cv.return_value = mock_instance

        train_evaluate_model(
            X_train_processed, y_train, X_test_processed, y_test,
            X_train_processed, X_test_processed, model_type='logistic_regression',
            param_grid={'classifier__C': LR_C_OPTIONS}
        )

        # Assert MLflow logging calls
        mock_log_param.assert_called_once_with('logistic_regression_classifier__C', 1.0)
        mock_log_metric.assert_any_call('logistic_regression_best_cv_roc_auc', 0.85)
        # Assert that accuracy is logged, but don't check the exact value due to randomness
        mock_log_metric.assert_any_call('logistic_regression_accuracy', ANY)
        mock_log_sklearn_model.assert_called_once()
        # Check input_example for sklearn models
        assert mock_log_sklearn_model.call_args[1]['input_example'].equals(X_train_processed[:5])

    # Test XGBoost logging separately due to different log_model call
    mock_log_param.reset_mock()
    mock_log_metric.reset_mock()
    mock_log_sklearn_model.reset_mock()
    mock_log_xgboost_model.reset_mock()

    with patch('model_training.RandomizedSearchCV') as mock_randomized_search_cv:
        mock_instance = MagicMock()
        mock_instance.best_estimator_ = MagicMock()
        mock_instance.best_estimator_.predict.side_effect = lambda X: np.random.randint(0, 2, len(X))
        mock_instance.best_estimator_.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
        mock_instance.best_score_ = 0.88
        mock_instance.best_params_ = {'classifier__n_estimators': 100}
        mock_randomized_search_cv.return_value = mock_instance

        train_evaluate_model(
            X_train_processed, y_train, X_test_processed, y_test,
            X_train_processed, X_test_processed, model_type='xgboost',
            param_grid={
                'classifier__n_estimators': XGB_N_ESTIMATORS_OPTIONS,
                'classifier__learning_rate': XGB_LEARNING_RATE_OPTIONS
            }
        )

        # Assert MLflow logging calls for XGBoost
        mock_log_param.assert_called_once_with('xgboost_classifier__n_estimators', 100)
        mock_log_metric.assert_any_call('xgboost_best_cv_roc_auc', 0.88)
        # Assert that accuracy is logged, but don't check the exact value due to randomness
        mock_log_metric.assert_any_call('xgboost_accuracy', ANY)
        mock_log_xgboost_model.assert_called_once()
        # Check input_example for xgboost models
        assert mock_log_xgboost_model.call_args[1]['input_example'].equals(X_train_processed[:5])