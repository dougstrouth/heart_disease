import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd # Import pandas for CategoricalDtype

# Import DASK_TYPE from config
from config import DASK_TYPE

# Conditionally import Dask-ML preprocessors
if DASK_TYPE != 'local':
    from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
    from dask_ml.preprocessing import OneHotEncoder as DaskOneHotEncoder
    from dask_ml.impute import SimpleImputer as DaskSimpleImputer
    from dask_ml.compose import ColumnTransformer as DaskColumnTransformer
    from sklearn.base import BaseEstimator, TransformerMixin # Import for custom transformer

    class ToCategoricalDtype(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            # Ensure X is a Dask DataFrame or Series
            if isinstance(X, pd.DataFrame):
                return X.astype('category')
            elif isinstance(X, pd.Series):
                return X.astype('category')
            elif hasattr(X, 'to_dask_dataframe'): # Dask Array
                return X.to_dask_dataframe().astype('category')
            else: # Dask DataFrame
                return X.astype('category')
else:
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

def get_preprocessor(categorical_features, numerical_features, binary_features):
    """
    Creates and returns a ColumnTransformer for preprocessing.
    Conditionally uses Dask-ML preprocessors if DASK_TYPE is not 'local'.
    """
    if DASK_TYPE != 'local':
        numerical_transformer = Pipeline(steps=[
            ('imputer', DaskSimpleImputer(strategy='mean')),
            ('scaler', DaskStandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', DaskSimpleImputer(strategy='constant', fill_value='missing')),
            ('to_category', ToCategoricalDtype()), # Use custom transformer
            ('onehot', DaskOneHotEncoder(handle_unknown='error'))
        ])
        binary_transformer = Pipeline(steps=[
            ('imputer', DaskSimpleImputer(strategy='most_frequent')),
            ('to_category', ToCategoricalDtype()),
            ('onehot', DaskOneHotEncoder(handle_unknown='error'))
        ])
        preprocessor_class = DaskColumnTransformer
    else:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ])
        preprocessor_class = ColumnTransformer

    preprocessor = preprocessor_class(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ],
        remainder='drop'
    )
    return preprocessor

def get_feature_names(preprocessor):
    """
    Gets feature names from a fitted ColumnTransformer.
    Handles both scikit-learn and Dask-ML ColumnTransformers.
    """
    feature_names = []
    for transformer_name, transformer, features in preprocessor.transformers_:
        if transformer_name == 'remainder' and transformer == 'drop':
            continue
        # Check if the transformer is a Pipeline and get the last step
        if isinstance(transformer, Pipeline):
            final_transformer = transformer.steps[-1][1]
        else:
            final_transformer = transformer

        if hasattr(final_transformer, 'get_feature_names_out'):
            names = final_transformer.get_feature_names_out(features)
            feature_names.extend(names)
        else:
            feature_names.extend(features)
    return feature_names

def preprocess_data(X_train, X_test, categorical_features, numerical_features, binary_features):
    """
    Applies preprocessing to training and testing data.
    Returns Dask Dataframes if DASK_TYPE is not 'local', otherwise NumPy arrays.
    """
    preprocessor = get_preprocessor(categorical_features, numerical_features, binary_features)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Only convert to numpy arrays if DASK_TYPE is 'local'
    if DASK_TYPE == 'local':
        if not isinstance(X_train_processed, np.ndarray):
            X_train_processed = X_train_processed.toarray()
        if not isinstance(X_test_processed, np.ndarray):
            X_test_processed = X_test_processed.toarray()

    processed_feature_names = get_feature_names(preprocessor)

    return X_train_processed, X_test_processed, processed_feature_names
