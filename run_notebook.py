import pandas as pd
import subprocess
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight

# Cell 1: Data Unification and Loading
new_data_path = '/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/johnsmith88/heart-disease-data/heart.csv'
try:
    new_data_content = subprocess.check_output(['cat', new_data_path]).decode('utf-8')
    df_new = pd.read_csv(StringIO(new_data_content))
    df_new['source'] = 'johnsmith88'
except Exception as e:
    print(f"Error loading new dataset: {e}")
    df_new = pd.DataFrame()

existing_data_path = '/Users/dougstrouth/Documents/datasets/kaggle_data_sets/data/pratyushpuri/heart-disease-dataset-3k-rows-python-code-2025/heart_disease_dataset.csv'
try:
    df_existing = pd.read_csv(existing_data_path)
    df_existing['source'] = 'pratyushpuri'
except FileNotFoundError:
    print(f"Error: The file was not found at {existing_data_path}")
    df_existing = pd.DataFrame()

if not df_new.empty and not df_existing.empty:
    df_new.rename(columns={'target': 'heart_disease'}, inplace=True)

    for col in ['smoking', 'diabetes', 'bmi']:
        if col not in df_new.columns:
            df_new[col] = 0

    existing_cols = [col for col in df_existing.columns if col != 'source']
    df_new = df_new[existing_cols + ['source']]

    df_unified = pd.concat([df_existing, df_new], ignore_index=True)

    unified_data_path = 'unified_heart_disease_dataset.csv'
    df_unified.to_csv(unified_data_path, index=False)
    print(f"Unified dataset created at {unified_data_path}")
    print(f"Unified dataset dimensions: {df_unified.shape}")

    df = pd.read_csv(unified_data_path)
    print("\nUnified dataset loaded for analysis.")

# Cell 2: Correlation Analysis
if 'df' in locals():
    correlation_matrix = df.corr(numeric_only=True)
    heart_disease_correlation = correlation_matrix['heart_disease'].sort_values(ascending=False)
    print("\nCorrelation of columns with heart_disease:")
    print(heart_disease_correlation)

# Cell 3: Random Forest Model
if 'df' in locals():
    X = df.drop(['heart_disease', 'source'], axis=1)
    y = df['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Model Accuracy: {accuracy}")

    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nRandom Forest Feature Importances:")
    print(feature_importances)

# Cell 4: Gradient Boosting Model with GridSearchCV
if 'df' in locals():
    X = df.drop(['heart_disease', 'source'], axis=1)
    y = df['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.7, 0.8]
    }

    gb_model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_gb_model = grid_search.best_estimator_

    y_pred = best_gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nBest Hyperparameters: {grid_search.best_params_}")
    print(f"New Model Accuracy (Gradient Boosting): {accuracy}")

    feature_importances = pd.Series(best_gb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nGradient Boosting Feature Importances:")
    print(feature_importances)

# Cell 5: Confusion Matrix and Misclassified Samples
if 'df' in locals() and 'best_gb_model' in locals():
    X = df.drop(['heart_disease', 'source'], axis=1)
    y = df['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    y_pred = best_gb_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_samples = X_test.iloc[misclassified_indices]
    misclassified_samples['true_label'] = y_test.iloc[misclassified_indices]
    misclassified_samples['predicted_label'] = y_pred[misclassified_indices]

    print("\nMisclassified Samples (first 5):")
    print(misclassified_samples.head())

    print("\nAnalysis of Misclassified Samples:")
    print(misclassified_samples.describe())

# Cell 6: Descriptive Statistics
if 'df' in locals():
    print("\nDescriptive Statistics of the Entire Dataset:")
    print(df.describe())

# Cell 7: Gradient Boosting with Class Weights
if 'df' in locals() and 'grid_search' in locals():
    X = df.drop(['heart_disease', 'source'], axis=1)
    y = df['heart_disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    gb_model_weighted = GradientBoostingClassifier(random_state=42, **grid_search.best_params_)
    gb_model_weighted.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred_weighted = gb_model_weighted.predict(X_test)
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
    cm_weighted = confusion_matrix(y_test, y_pred_weighted)

    print(f"\nNew Model Accuracy (with class weights): {accuracy_weighted}")
    print("\nNew Confusion Matrix (with class weights):")
    print(cm_weighted)

# Cell 8: Visualizations
if 'df' in locals():
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Key Features by Heart Disease', fontsize=16)

    sns.histplot(data=df, x='age', hue='heart_disease', multiple='stack', ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution')

    sns.histplot(data=df, x='chol', hue='heart_disease', multiple='stack', ax=axes[0, 1])
    axes[0, 1].set_title('Cholesterol Distribution')

    sns.histplot(data=df, x='thalach', hue='heart_disease', multiple='stack', ax=axes[1, 0])
    axes[1, 0].set_title('Max Heart Rate Distribution')

    sns.histplot(data=df, x='bmi', hue='heart_disease', multiple='stack', ax=axes[1, 1])
    axes[1, 1].set_title('BMI Distribution')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Categorical Features vs. Heart Disease', fontsize=16)

    sns.countplot(data=df, x='cp', hue='heart_disease', ax=axes[0, 0])
    axes[0, 0].set_title('Chest Pain Type vs. Heart Disease')
    axes[0, 0].set_xticklabels(['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])

    sns.countplot(data=df, x='sex', hue='heart_disease', ax=axes[0, 1])
    axes[0, 1].set_title('Sex vs. Heart Disease')
    axes[0, 1].set_xticklabels(['Female', 'Male'])

    sns.countplot(data=df, x='smoking', hue='heart_disease', ax=axes[1, 0])
    axes[1, 0].set_title('Smoking vs. Heart Disease')
    axes[1, 0].set_xticklabels(['No', 'Yes'])

    sns.countplot(data=df, x='diabetes', hue='heart_disease', ax=axes[1, 1])
    axes[1, 1].set_title('Diabetes vs. Heart Disease')
    axes[1, 1].set_xticklabels(['No', 'Yes'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()