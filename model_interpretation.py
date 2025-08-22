import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from config import SHOW_PLOTS

def interpret_model(model_pipeline, X_train_processed, feature_names):
    """
    Interprets the model, focusing on feature importances for tree-based models,
    coefficients for linear models, and SHAP values for deeper insights.

    Args:
        model_pipeline: The trained model pipeline (containing the classifier).
        X_train_processed: The preprocessed training data (NumPy array or DataFrame).
        feature_names: A list of feature names corresponding to the columns of X_train_processed.
    """
    if model_pipeline is None:
        print("Cannot interpret: Model pipeline is None.")
        return None

    classifier = model_pipeline.named_steps['classifier']
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)

    if hasattr(classifier, 'feature_importances_'):
        print("\n--- Feature Importances (Tree-based Model) ---")
        importances = classifier.feature_importances_
        feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
        print(feature_importances_df.head(10))

        if SHOW_PLOTS:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importances_df.head(15))
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

        print("\n--- SHAP Values (Tree-based Model) ---")
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_sample = X_train_df.sample(n=min(1000, X_train_df.shape[0]), random_state=42)
            shap_values = explainer.shap_values(shap_sample)

            if SHOW_PLOTS:
                shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False)
                plt.title('SHAP Feature Importance (Overall)')
                plt.tight_layout()
                plt.show()

                if isinstance(shap_values, list): # For binary classification
                    shap.summary_plot(shap_values[1], shap_sample, show=False)
                else:
                    shap.summary_plot(shap_values, shap_sample, show=False)
                plt.title('SHAP Summary Plot (Impact and Direction)')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error generating SHAP plots for tree model: {e}")

        return feature_importances_df

    elif hasattr(classifier, 'coef_'):
        print("\n--- Feature Coefficients (Linear Model) ---")
        coefficients = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
        feature_coefficients_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
        feature_coefficients_df['abs_coefficient'] = np.abs(feature_coefficients_df['coefficient'])
        feature_coefficients_df = feature_coefficients_df.sort_values(by='abs_coefficient', ascending=False)
        print(feature_coefficients_df.head(10))

        if SHOW_PLOTS:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='coefficient', y='feature', data=feature_coefficients_df.head(15))
            plt.title('Top 15 Feature Coefficients (Absolute Value)')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

        print("\n--- SHAP Values (Linear Model) ---")
        try:
            background_sample = X_train_df.sample(n=min(100, X_train_df.shape[0]), random_state=42)
            explainer = shap.KernelExplainer(classifier.predict_proba, background_sample)
            shap_sample = X_train_df.sample(n=min(1000, X_train_df.shape[0]), random_state=42)
            shap_values = explainer.shap_values(shap_sample)

            if SHOW_PLOTS:
                shap.summary_plot(shap_values[1], shap_sample, show=False)
                plt.title('SHAP Summary Plot (Impact and Direction)')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error generating SHAP plots for linear model: {e}")

        return feature_coefficients_df

    else:
        print("Model does not have feature importances or coefficients. Consider SHAP/LIME.")
        return None