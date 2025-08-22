import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import get_preprocessor
from ensemble_utils import train_stacked_model
from config import META_CLASSIFIER

# Define dummy data for testing
@pytest.fixture
def dummy_data():
    # Create a simple DataFrame that mimics the structure of your combined_df
    # Ensure it has numerical, categorical, and binary features as defined in config.py
    data = {
        'age': np.random.randint(20, 80, 100),
        'sex': np.random.choice([0, 1], 100),
        'cp': np.random.choice([0, 1, 2, 3], 100),
        'trestbps': np.random.randint(100, 180, 100),
        'chol': np.random.randint(150, 300, 100),
        'fbs': np.random.choice([0, 1], 100),
        'restecg': np.random.choice([0, 1, 2], 100),
        'thalach': np.random.randint(100, 200, 100),
        'exang': np.random.choice([0, 1], 100),
        'oldpeak': np.random.rand(100) * 5,
        'slope': np.random.choice([0, 1, 2], 100), # Added slope column
        'ca': np.random.randint(0, 4, 100),
        'thal': np.random.choice([3, 6, 7], 100),
        'bmi': np.random.rand(100) * 10 + 20, # Dummy BMI
        'smoking': np.random.choice([0, 1], 100), # Dummy Smoking
        'diabetes': np.random.choice([0, 1], 100), # Dummy Diabetes
        'heart_disease': np.random.randint(0, 2, 100) # Target variable
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def processed_data_and_models(dummy_data):
    X = dummy_data.drop('heart_disease', axis=1)
    y = dummy_data['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define features based on your config.py (assuming these are consistent)
    NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'bmi']
    CATEGORICAL_FEATURES = ['sex', 'cp', 'restecg', 'slope', 'thal']
    BINARY_FEATURES = ['fbs', 'exang', 'smoking', 'diabetes']

    preprocessor = get_preprocessor(CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES)

    # Fit the preprocessor on the full training data to learn all categories
    preprocessor.fit(X_train)
    # Transform a dummy row to get the number of features after transformation
    expected_features = preprocessor.transform(X_train.head(1)).shape[1]

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train dummy base models
    lr_model = Pipeline(steps=[('classifier', LogisticRegression(solver='liblinear', random_state=42))])
    rf_model = Pipeline(steps=[('classifier', RandomForestClassifier(random_state=42))])
    xgb_model = Pipeline(steps=[('classifier', XGBClassifier(eval_metric='logloss', random_state=42))])

    lr_model.fit(X_train_processed, y_train)
    rf_model.fit(X_train_processed, y_train)
    xgb_model.fit(X_train_processed, y_train)

    base_models = {'lr': lr_model, 'rf': rf_model, 'xgb': xgb_model}

    # Train stacked model and get stacked_input_example
    stacked_model, _, _, _, stacked_input_example = train_stacked_model(
        base_models, X_train_processed, y_train, X_test_processed, y_test, META_CLASSIFIER
    )

    return X_train_processed, X_test_processed, stacked_input_example, preprocessor, expected_features

def test_processed_data_feature_count(processed_data_and_models):
    X_train_processed, X_test_processed, _, _, expected_features = processed_data_and_models
    assert X_train_processed.shape[1] == expected_features
    assert X_test_processed.shape[1] == expected_features

def test_stacked_model_input_feature_count(processed_data_and_models):
    _, _, stacked_input_example, _, _ = processed_data_and_models
    # The stacked model input should have features equal to the number of base models
    # In this case, 3 (LR, RF, XGB)
    expected_stacked_features = 3
    assert stacked_input_example.shape[1] == expected_stacked_features

    # Also check that the data types are float, as expected by LogisticRegression
    assert all(np.issubdtype(dtype, np.floating) for dtype in stacked_input_example.dtypes)
