import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration options
from config import SHOW_PLOTS
from data_utils import ALL_NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES

def interpret_model(model_pipeline, preprocessor, numerical_features, categorical_features):
    """
    Interprets the model, focusing on feature importances for tree-based models
    and coefficients for linear models.
    """
    if model_pipeline is None:
        print("Cannot interpret: Model pipeline is None.")
        return None

    classifier = model_pipeline.named_steps['classifier']
    fitted_preprocessor = model_pipeline.named_steps['preprocessor']

    # Get feature names after preprocessing
    processed_numerical_features = [f for f in numerical_features if f in fitted_preprocessor.transformers_[0][2]]
    processed_categorical_features = [f for f in categorical_features if f in fitted_preprocessor.transformers_[1][2]]

    all_feature_names = []
    # Get names from numerical features (after scaling)
    all_feature_names.extend(processed_numerical_features)

    # Get names from one-hot encoded categorical features
    ohe_transformer = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
    if hasattr(ohe_transformer, 'get_feature_names_out'):
        ohe_feature_names = ohe_transformer.get_feature_names_out()
    else: # Fallback for older sklearn versions
        ohe_feature_names = ohe_transformer.get_feature_names(processed_categorical_features)
    all_feature_names.extend(list(ohe_feature_names))


    if hasattr(classifier, 'feature_importances_'):
        print("\n--- Feature Importances (Tree-based Model) ---")
        importances = classifier.feature_importances_
        feature_importances_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
        print(feature_importances_df.head(10))

        # Plotting feature importances
        if SHOW_PLOTS:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importances_df.head(15))
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()
        return feature_importances_df

    elif hasattr(classifier, 'coef_'):
        print("\n--- Feature Coefficients (Linear Model) ---")
        # For binary classification, coef_ is usually 1D array
        coefficients = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
        feature_coefficients_df = pd.DataFrame({'feature': all_feature_names, 'coefficient': coefficients})
        feature_coefficients_df['abs_coefficient'] = np.abs(feature_coefficients_df['coefficient'])
        feature_coefficients_df = feature_coefficients_df.sort_values(by='abs_coefficient', ascending=False)
        print(feature_coefficients_df.head(10))

        # Plotting coefficients (optional, similar to feature importances)
        if SHOW_PLOTS:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='coefficient', y='feature', data=feature_coefficients_df.head(15))
            plt.title('Top 15 Feature Coefficients (Absolute Value)')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()
        return feature_coefficients_df

    else:
        print("Model does not have feature importances or coefficients. Consider SHAP/LIME.")
        return None
