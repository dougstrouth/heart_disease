import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from config import DASK_TYPE

# Import Dask-ML preprocessors
if DASK_TYPE == 'coiled':
    from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
    from dask_ml.preprocessing import OneHotEncoder as DaskOneHotEncoder

def get_preprocessor(categorical_features, numerical_features, binary_features):
    # logger = logging.getLogger('heart_disease_analysis') # Removed unused logger assignment
    """
    Creates and returns a ColumnTransformer for preprocessing.
    Conditionally uses Dask-ML preprocessors if DASK_TYPE is 'coiled'.
    """
    if DASK_TYPE == 'coiled':
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), # SimpleImputer is pandas/numpy compatible
            ('scaler', DaskStandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', DaskOneHotEncoder(handle_unknown='ignore'))
        ])
    else:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

    binary_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ],
        remainder='drop'
    )
    return preprocessor

def get_feature_names(preprocessor):
    # logger = logging.getLogger('heart_disease_analysis') # Removed unused logger assignment
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
            # Dask-ML OneHotEncoder has get_feature_names_out
            names = final_transformer.get_feature_names_out(features)
            # For Dask-ML OneHotEncoder, get_feature_names_out returns a Dask Array, compute it
            if DASK_TYPE == 'coiled' and isinstance(names, np.ndarray) and names.dtype == object:
                feature_names.extend(names.tolist())
            else:
                feature_names.extend(names)
        else:
            feature_names.extend(features)
    return feature_names