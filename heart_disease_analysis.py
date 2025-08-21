# To install xgboost: pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier # Import XGBClassifier

# --- Configuration ---
SHOW_PLOTS = False  # Set to True to display plots, False to suppress them

# --- Data Loading Function ---
def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None

# --- Data Harmonization Function ---
def harmonize_datasets(df_synthetic, df_uci):
    """
    Harmonizes two heart disease datasets for combination.
    Aligns column names, re-encodes 'thal', handles unique features, and adds source tracking.
    """
    if df_synthetic is None or df_uci is None:
        print("Cannot harmonize: One or both DataFrames are None.")
        return None

    print("\n--- Harmonizing Datasets ---")

    # Make copies to avoid modifying original DataFrames
    df_synthetic_harmonized = df_synthetic.copy()
    df_uci_harmonized = df_uci.copy()

    print("\n--- Debugging: Initial Columns ---")
    print("Synthetic DF Columns:", df_synthetic_harmonized.columns.tolist())
    print("UCI DF Columns:", df_uci_harmonized.columns.tolist())

    # 1. Rename columns in UCI dataset to match synthetic dataset
    # Correcting 'thalch' to 'thalach'
    if 'thalch' in df_uci_harmonized.columns:
        df_uci_harmonized.rename(columns={'thalch': 'thalach'}, inplace=True)
        print("Renamed 'thalch' to 'thalach' in UCI dataset.")

    # Correcting target column from 'num' to 'heart_disease'
    if 'num' in df_uci_harmonized.columns:
        df_uci_harmonized.rename(columns={'num': 'heart_disease'}, inplace=True)
        print("Renamed 'num' to 'heart_disease' in UCI dataset.")

    # 2. Re-encode 'thal' column in UCI dataset
    # Synthetic thal: 3=normal, 6=fixed defect, 7=reversible defect
    # UCI thal: 0=normal, 1=fixed defect, 2=reversible defect
    thal_mapping_uci_to_synthetic = {0: 3, 1: 6, 2: 7}
    if 'thal' in df_uci_harmonized.columns:
        # Convert thal to numeric first to handle potential non-numeric values before mapping
        df_uci_harmonized['thal'] = pd.to_numeric(df_uci_harmonized['thal'], errors='coerce')
        df_uci_harmonized['thal'] = df_uci_harmonized['thal'].map(thal_mapping_uci_to_synthetic)
        print("Re-encoded 'thal' column in UCI dataset.")

    # 3. Add 'source' column to both datasets
    if 'source' not in df_synthetic_harmonized.columns:
        df_synthetic_harmonized['source'] = 'Synthetic' # Assuming original synthetic data doesn't have it
    print("Ensured 'source' column in synthetic dataset.")

    df_uci_harmonized['source'] = 'UCI'
    print("Added 'source' column to UCI dataset.")

    # 4. Handle features unique to synthetic dataset: 'smoking', 'diabetes', 'bmi'
    unique_synthetic_features = ['smoking', 'diabetes', 'bmi']
    for feature in unique_synthetic_features:
        if feature not in df_uci_harmonized.columns:
            df_uci_harmonized[feature] = 0  # Impute with 0
            print(f"Added and imputed '{feature}' with 0 for UCI dataset.")

    print("\n--- Debugging: Columns after initial harmonization steps ---")
    print("Synthetic DF Harmonized Columns:", df_synthetic_harmonized.columns.tolist())
    print("UCI DF Harmonized Columns:", df_uci_harmonized.columns.tolist())

    # Ensure both dataframes have the exact same columns in the same order before concatenation
    # Get all columns from the synthetic dataset (which now includes 'source' and original unique features)
    final_columns = df_synthetic_harmonized.columns.tolist()
    print("\n--- Debugging: Final Columns for Alignment ---")
    print("Final Columns:", final_columns)

    # Reorder UCI columns to match synthetic dataset's column order
    df_uci_harmonized = df_uci_harmonized[final_columns].copy()

    print("Datasets harmonized successfully. Ready for concatenation.")
    return df_synthetic_harmonized, df_uci_harmonized

# --- Exploratory Data Analysis (EDA) Function ---
def perform_eda(df, dataset_name, numerical_features, categorical_features):
    """
    Performs basic Exploratory Data Analysis (EDA) on the DataFrame.
    Prints head, info, describe, missing values, and target distribution.
    Generates and displays basic plots.
    """
    if df is None:
        print(f"Cannot perform EDA: {dataset_name} DataFrame is None.")
        return

    print(f"\n--- EDA for {dataset_name} ---")
    print("Head:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nDescription:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())

    target_column = 'heart_disease' # Now consistent across harmonized datasets
    if target_column in df.columns:
        print(f"\nTarget distribution ({target_column}):")
        print(df[target_column].value_counts(normalize=True))

        # Visualize target distribution
        if SHOW_PLOTS:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=target_column, data=df)
            plt.title(f'Distribution of {target_column} ({dataset_name})')
            plt.show()

    # Visualize numerical features distributions
    if numerical_features and all(col in df.columns for col in numerical_features):
        if SHOW_PLOTS:
            df[numerical_features].hist(bins=15, figsize=(15, 10))
            plt.suptitle(f'Histograms of Numerical Features ({dataset_name})')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    # Visualize categorical features distributions
    if categorical_features:
        for col in categorical_features:
            if col in df.columns:
                if SHOW_PLOTS:
                    plt.figure(figsize=(6, 4))
                    sns.countplot(x=col, data=df)
                    plt.title(f'Distribution of {col} ({dataset_name})')
                    plt.show()

# --- Data Preprocessing Function ---
def preprocess_data(df, target_column, categorical_features, numerical_features):
    """
    Preprocesses the DataFrame by separating features and target,
    handling categorical feature types, and setting up a ColumnTransformer.
    """
    if df is None:
        print("Cannot preprocess: DataFrame is None.")
        return None, None, None

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Ensure identified categorical features are treated as such
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')

    # Preprocessing pipelines for numerical and categorical features
    # Exclude 'source' column from preprocessing as it's for tracking, not modeling
    features_for_preprocessing = [f for f in X.columns if f != 'source']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Impute missing numerical values
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, [f for f in numerical_features if f in features_for_preprocessing]),
            ('cat', categorical_transformer, [f for f in categorical_features if f in features_for_preprocessing])
        ])

    print("\nPreprocessing setup complete.")
    return X, y, preprocessor

# --- Model Training and Evaluation Function ---
def train_evaluate_model(X, y, preprocessor, model_type='logistic_regression', param_grid=None):
    """
    Trains and evaluates a specified machine learning model.
    If param_grid is provided, performs GridSearchCV for hyperparameter tuning.
    """
    if X is None or y is None or preprocessor is None:
        print("Cannot train/evaluate: Data or preprocessor is None.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\nData split into training and testing sets.")

    if model_type == 'logistic_regression':
        classifier = LogisticRegression(solver='liblinear', random_state=42)
        model_name = "Logistic Regression"
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
        model_name = "Random Forest"
    elif model_type == 'xgboost': # Add XGBoost
        classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model_name = "XGBoost"
    else:
        print(f"Error: Unknown model type '{model_type}'.")
        return None, None, None

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    if param_grid:
        print(f"\nPerforming GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
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

    # Evaluate the model
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

    return best_model, y_pred, y_proba, metrics

# --- Model Interpretability Function ---
def interpret_model(model_pipeline, preprocessor, numerical_features, categorical_features):
    """
    Interprets the model, focusing on feature importances for tree-based models.
    """
    if model_pipeline is None:
        print("Cannot interpret: Model pipeline is None.")
        return None

    classifier = model_pipeline.named_steps['classifier']
    # Access the fitted preprocessor from the model_pipeline
    fitted_preprocessor = model_pipeline.named_steps['preprocessor']

    if hasattr(classifier, 'feature_importances_'):
        print("\n--- Feature Importances ---")
        # Get feature names after one-hot encoding
        # Need to get feature names from the preprocessor after it has been fitted
        # The preprocessor in the pipeline is fitted during model_pipeline.fit()
        # We need to ensure we only get names for features that were actually processed

        processed_numerical_features = [f for f in numerical_features if f in fitted_preprocessor.transformers_[0][2]]
        processed_categorical_features = [f for f in categorical_features if f in fitted_preprocessor.transformers_[1][2]]

        ohe_transformer = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
        if hasattr(ohe_transformer, 'get_feature_names_out'):
            ohe_feature_names = ohe_transformer.get_feature_names_out(processed_categorical_features)
        else: # Fallback for older sklearn versions
            ohe_feature_names = ohe_transformer.get_feature_names(processed_categorical_features)

        all_feature_names = processed_numerical_features + list(ohe_feature_names)

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

        print("\nFurther interpretability can be achieved using SHAP or LIME libraries.")
        return feature_importances_df
    else:
        print("Model does not have feature importances (e.g., Logistic Regression). Consider SHAP/LIME.")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    start_time_total = time.time()
    print("\n--- Starting Heart Disease Analysis ---")

    # Define feature lists based on the full set of features after harmonization
    # This includes features unique to synthetic dataset, which are imputed for UCI
    ALL_NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'bmi']
    ALL_CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'smoking', 'diabetes']
    TARGET_COLUMN = 'heart_disease'

    # 1. Load Data
    start_time_load = time.time()
    df_synthetic = load_data('unified_heart_disease_dataset.csv')
    df_uci = load_data('/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/edwankarimsony/heart-disease-data/heart_disease_uci.csv')
    end_time_load = time.time()
    print(f"Data Loading completed in {end_time_load - start_time_load:.2f} seconds.")

    # 2. Harmonize and Combine Datasets
    start_time_harmonize = time.time()
    combined_df = None
    if df_synthetic is not None and df_uci is not None:
        df_synthetic_harmonized, df_uci_harmonized = harmonize_datasets(df_synthetic, df_uci)
        if df_synthetic_harmonized is not None and df_uci_harmonized is not None:
            combined_df = pd.concat([df_synthetic_harmonized, df_uci_harmonized], ignore_index=True)
            print(f"\nCombined dataset created with {len(combined_df)} rows.")
            print("Combined dataset head:")
            print(combined_df.head())
            print("Combined dataset info:")
            print(combined_df.info())
            print("Combined dataset description:")
            print(combined_df.describe())
            print("Combined dataset missing values:")
            print(combined_df.isnull().sum())

            # Drop rows where the target column (heart_disease) is NaN
            initial_rows = len(combined_df)
            combined_df.dropna(subset=[TARGET_COLUMN], inplace=True)
            rows_after_dropna = len(combined_df)
            print(f"Dropped {initial_rows - rows_after_dropna} rows with NaN in '{TARGET_COLUMN}'.")
            print(f"Combined dataset now has {rows_after_dropna} rows.")

            # Convert all categorical features to string type and fill NaNs
            for col in ALL_CATEGORICAL_FEATURES:
                if col in combined_df.columns:
                    # Convert to string, coercing errors will turn unconvertible values into NaN
                    combined_df[col] = combined_df[col].astype(str)
                    # Fill any NaNs that might have resulted from coercion or were already present
                    combined_df[col].fillna('missing', inplace=True)
                    print(f"Converted '{col}' to string and filled NaNs.")

            # Ensure heart_disease is integer type for classification
            # Note: The UCI dataset has values 0, 1, 2, 3, 4 for heart_disease. 
            # For binary classification, these typically need to be binarized (e.g., >0 means disease).
            # For now, we'll keep them as is, but this is a point for future refinement.
            combined_df[TARGET_COLUMN] = combined_df[TARGET_COLUMN].astype(int)
            print(f"Converted '{TARGET_COLUMN}' to integer type.")

            # Binarize the target variable: 0 = no disease, >0 = disease
            # This is crucial because the UCI dataset has 0-4 values for heart_disease
            combined_df[TARGET_COLUMN] = (combined_df[TARGET_COLUMN] > 0).astype(int)
            print(f"Binarized '{TARGET_COLUMN}': values > 0 converted to 1.")
    end_time_harmonize = time.time()
    print(f"Data Harmonization and Combination completed in {end_time_harmonize - start_time_harmonize:.2f} seconds.")

    # 3. Perform EDA on Combined Dataset
    start_time_eda = time.time()
    if combined_df is not None:
        perform_eda(combined_df, "Combined Dataset", ALL_NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES)
    end_time_eda = time.time()
    print(f"EDA completed in {end_time_eda - start_time_eda:.2f} seconds.")

    # 4. Preprocess Combined Data
    start_time_preprocess = time.time()
    # Exclude 'source' column from features for modeling
    features_for_modeling_numerical = ALL_NUMERICAL_FEATURES
    features_for_modeling_categorical = ALL_CATEGORICAL_FEATURES + ['source'] # Add source to categorical for preprocessing

    X_combined, y_combined, preprocessor_combined = preprocess_data(
        combined_df, TARGET_COLUMN, features_for_modeling_categorical, features_for_modeling_numerical
    )
    end_time_preprocess = time.time()
    print(f"Data Preprocessing completed in {end_time_preprocess - start_time_preprocess:.2f} seconds.")

    # 5. Train and Evaluate Models on Combined Data
    start_time_train = time.time()
    if X_combined is not None:
        # Define parameter grids for GridSearchCV
        param_grid_lr = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0], # Expanded C values
            'classifier__solver': ['liblinear', 'lbfgs']
        }

        param_grid_rf = {
            'classifier__n_estimators': [150, 200, 250], # More estimators
            'classifier__max_features': ['sqrt', 'log2', 0.8], # Add a float for max_features
            'classifier__max_depth': [15, 20, 25, None], # Refine around 20, add None
            'classifier__min_samples_split': [2, 4, 6], # More granular
            'classifier__min_samples_leaf': [1, 2, 3] # Add min_samples_leaf
        }

        param_grid_xgb = {
            'classifier__n_estimators': [150, 200, 250],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [4, 5, 6],
            'classifier__subsample': [0.7, 0.8, 1.0],
            'classifier__colsample_bytree': [0.7, 0.8, 1.0],
            'classifier__gamma': [0, 0.1, 0.2] # Add gamma
        }

        # Logistic Regression with GridSearchCV
        start_time_lr = time.time()
        lr_model_combined, lr_y_pred_combined, lr_y_proba_combined, lr_metrics_combined = train_evaluate_model(
            X_combined, y_combined, preprocessor_combined, model_type='logistic_regression', param_grid=param_grid_lr
        )
        end_time_lr = time.time()
        print(f"Logistic Regression training and evaluation completed in {end_time_lr - start_time_lr:.2f} seconds.")

        # Random Forest with GridSearchCV
        start_time_rf = time.time()
        rf_model_combined, rf_y_pred_combined, rf_y_proba_combined, rf_metrics_combined = train_evaluate_model(
            X_combined, y_combined, preprocessor_combined, model_type='random_forest', param_grid=param_grid_rf
        )
        end_time_rf = time.time()
        print(f"Random Forest training and evaluation completed in {end_time_rf - start_time_rf:.2f} seconds.")

        # XGBoost with GridSearchCV
        start_time_xgb = time.time()
        xgb_model_combined, xgb_y_pred_combined, xgb_y_proba_combined, xgb_metrics_combined = train_evaluate_model(
            X_combined, y_combined, preprocessor_combined, model_type='xgboost', param_grid=param_grid_xgb
        )
        end_time_xgb = time.time()
        print(f"XGBoost training and evaluation completed in {end_time_xgb - start_time_xgb:.2f} seconds.")

    end_time_train = time.time()
    print(f"Model Training and Evaluation completed in {end_time_train - start_time_train:.2f} seconds.")

    # 6. Model Interpretability (for Random Forest on Combined Data)
    start_time_interpret = time.time()
    # Note: XGBoost also has feature importances, but accessing them is similar to RF.
    if rf_model_combined is not None:
        rf_feature_importances_combined = interpret_model(
            rf_model_combined, preprocessor_combined, features_for_modeling_numerical, features_for_modeling_categorical
        )

    # You can also interpret XGBoost model if needed
    # if xgb_model_combined is not None:
    #     xgb_feature_importances_combined = interpret_model(
    #         xgb_model_combined, preprocessor_combined, features_for_modeling_numerical, features_for_modeling_categorical
    #     )
    end_time_interpret = time.time()
    print(f"Model Interpretability completed in {end_time_interpret - start_time_interpret:.2f} seconds.")

    end_time_total = time.time()
    print(f"\n--- Analysis Complete on Combined Dataset --- Total time: {end_time_total - start_time_total:.2f} seconds.")
    print("\nRecommendations for further speed improvement:")
    print("1. GridSearchCV: The most time-consuming part is often hyperparameter tuning with GridSearchCV.")
    print("   - Reduce the size of parameter grids (fewer values per parameter).")
    print("   - Reduce the 'cv' (cross-validation) folds.")
    print("   - Consider RandomizedSearchCV instead of GridSearchCV for large search spaces.")
    print("2. Data Size: For very large datasets, consider sampling or using distributed computing frameworks (e.g., Dask).")
    print("3. Feature Engineering/Selection: Reduce the number of features if many are redundant or irrelevant.")
    print("4. Progress Bars: For long-running loops or processes, consider using 'tqdm' for visual progress indication.")
    print("   (e.g., 'from tqdm import tqdm' and wrap iterables like 'for item in tqdm(my_list):')")
    print("   However, integrating tqdm directly into GridSearchCV's internal process is more complex.")
    print("5. Caching: If you run the script multiple times with the same data, save preprocessed data (X, y) to disk.")
    print("   (e.g., using joblib.dump and joblib.load) to avoid re-running preprocessing.")
