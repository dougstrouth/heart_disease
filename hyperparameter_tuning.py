import mlflow
import optuna
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from dask_data_utils import get_processed_data as dask_get_processed_data
from pandas_data_utils import get_processed_data as pandas_get_processed_data
from data_schema import HeartDiseaseSchema
from preprocessing import get_preprocessor
from model_training import train_evaluate_model
from config import DASK_TYPE, LOCAL_UCI_PATH, LOCAL_SYNTHETIC_PATH, LOCAL_JOHNSMITH_PATH, GCS_DATA_PATH
from dask_utils import get_dask_client

def objective(trial, dask_client):
    with mlflow.start_run():
        # Log the trial number
        mlflow.log_param("trial_number", trial.number)

        # Define search space
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 1000)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)

        mlflow.log_params({
            "n_estimators": rf_n_estimators,
            "max_depth": rf_max_depth,
            "min_samples_split": rf_min_samples_split,
            "min_samples_leaf": rf_min_samples_leaf,
        })

        # Load and process data
        if DASK_TYPE in ['coiled', 'cloud']:
            dtype_spec = {'ca': 'object', 'cp': 'object', 'restecg': 'object', 'sex': 'object', 'slope': 'object', 'thal': 'object'}
            processed_df = dd.read_csv(GCS_DATA_PATH, dtype=dtype_spec, na_values=['?'])
            processed_df = processed_df.drop_duplicates()
        elif DASK_TYPE == 'local':
            processed_df = dask_get_processed_data(
                uci_path=LOCAL_UCI_PATH,
                synthetic_path=LOCAL_SYNTHETIC_PATH,
                johnsmith_path=LOCAL_JOHNSMITH_PATH
            )
        else: # 'pandas'
            processed_df = pandas_get_processed_data(
                uci_path=LOCAL_UCI_PATH,
                synthetic_path=LOCAL_SYNTHETIC_PATH,
                johnsmith_path=LOCAL_JOHNSMITH_PATH
            )

        X = processed_df.drop(HeartDiseaseSchema.TARGET_COLUMN, axis=1)
        y = processed_df[HeartDiseaseSchema.TARGET_COLUMN]

        if hasattr(X, 'compute'):
            X_pd = X.compute()
            y_pd = y.compute()
        else:
            X_pd = X
            y_pd = y

        X_train, X_test, y_train, y_test = train_test_split(X_pd, y_pd, test_size=0.2, random_state=42, stratify=y_pd if y_pd.value_counts().min() > 1 else None)

        binary_features = ['sex', 'fbs', 'exang', 'smoking', 'diabetes']
        preprocessor = get_preprocessor(HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE(), HeartDiseaseSchema.NUMERICAL_COLUMNS(), binary_features, use_dask_ml=False)

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Train and evaluate model
        param_grid = {
            'classifier__n_estimators': [rf_n_estimators],
            'classifier__max_depth': [rf_max_depth],
            'classifier__min_samples_split': [rf_min_samples_split],
            'classifier__min_samples_leaf': [rf_min_samples_leaf]
        }

        model, _, _, metrics = train_evaluate_model(
            X_train_processed, y_train, X_test_processed, y_test, 
            X_train_processed, X_test_processed, 
            model_type='random_forest', 
            param_grid=param_grid, 
            dask_client=dask_client
        )

        roc_auc = metrics['roc_auc']
        mlflow.log_metric("roc_auc", roc_auc)

        return roc_auc

def run_tuning(tuning_config):
    dask_client = get_dask_client(cluster_type=DASK_TYPE)
    try:
        study = optuna.create_study(direction="maximize")
        
        n_trials = 100  # Default number of trials
        timeout_seconds = None

        if DASK_TYPE == 'cloud' and tuning_config.get('cloud_run_mode') == 'test':
            print("Cloud run detected in 'test' mode. Running a single trial to verify functionality.")
            n_trials = 1
        
        if DASK_TYPE == 'coiled':
            max_time_hours = tuning_config.get('coiled_max_time_hours')
            if max_time_hours:
                timeout_seconds = max_time_hours * 3600
                print(f"Coiled run detected. Setting timeout to {max_time_hours} hour(s) ({timeout_seconds} seconds).")

        study.optimize(
            lambda trial: objective(trial, dask_client),
            n_trials=n_trials,
            timeout=timeout_seconds
        )

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    finally:
        if dask_client:
            dask_client.close()
            if hasattr(dask_client.cluster, "shutdown"):
                dask_client.cluster.shutdown()

    return study.best_trial.params


if __name__ == "__main__":
    run_tuning()