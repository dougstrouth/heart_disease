
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def get_preprocessor(categorical_features, numerical_features, binary_features):
    """
    Creates and returns a ColumnTransformer for preprocessing.
    """
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
    """
    Gets feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    for transformer_name, transformer, features in preprocessor.transformers_:
        if transformer_name == 'remainder' and transformer == 'drop':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(features)
            feature_names.extend(names)
        else:
            feature_names.extend(features)
    return feature_names
