import time
import dask.dataframe as dd


from pandas_data_utils import get_processed_data as pandas_get_processed_data, perform_eda
from dask_data_utils import get_processed_data as dask_get_processed_data
from data_schema import HeartDiseaseSchema
from utils.logger_config import setup_logging
from modeling import run_model_pipeline, perform_final_model_evaluation, log_analysis_results
from dask_utils import get_dask_client
import mlflow
from config import DASK_TYPE, SHOW_PLOTS, VERBOSE_OUTPUT, RUN_MODELING, META_CLASSIFIER

def run_analysis():
    """Main function to run the data analysis pipeline."""
    logger = setup_logging()
    logger.info("--- Starting Heart Disease Analysis ---")
    start_time = time.time()

    if DASK_TYPE in ['coiled', 'cloud']:
        from config import GCS_DATA_PATH
        logger.info(f"Using Dask to process data from GCS: {GCS_DATA_PATH}")
        # Add dtype specification from previous step to handle mixed types
        dtype_spec = {'ca': 'object', 'cp': 'object', 'restecg': 'object', 'sex': 'object', 'slope': 'object', 'thal': 'object'}
        processed_df = dd.read_csv(GCS_DATA_PATH, dtype=dtype_spec, na_values=['?'])
    elif DASK_TYPE == 'local':
        logger.info("Using Dask to process data from local files.")
        # Define file paths
        uci_path = 'tests/test_data/dummy_uci_data.csv'
        synthetic_path = 'tests/test_data/dummy_synthetic_data.csv'
        johnsmith_path = 'tests/test_data/dummy_johnsmith_data.csv'
        processed_df = dask_get_processed_data(
            uci_path=uci_path,
            synthetic_path=synthetic_path,
            johnsmith_path=johnsmith_path
        )
    else: # 'pandas'
        logger.info("Using Pandas to process data from local files.")
        # Define file paths
        uci_path = 'tests/test_data/dummy_uci_data.csv'
        synthetic_path = 'tests/test_data/dummy_synthetic_data.csv'
        johnsmith_path = 'tests/test_data/dummy_johnsmith_data.csv'
        processed_df = pandas_get_processed_data(
            uci_path=uci_path,
            synthetic_path=synthetic_path,
            johnsmith_path=johnsmith_path
        )

    logger.info(f"Data processing completed in {time.time() - start_time:.2f} seconds.")

    if processed_df is not None:
        # If using dask, compute the dataframe for EDA
        if isinstance(processed_df, dd.DataFrame):
                        perform_eda(processed_df.compute(), "Combined Dataset", HeartDiseaseSchema.NUMERICAL_COLUMNS(), HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE(), show_plots=SHOW_PLOTS, verbose_output=VERBOSE_OUTPUT)
        else:
            perform_eda(processed_df, "Combined Dataset", HeartDiseaseSchema.NUMERICAL_COLUMNS(), HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE(), show_plots=SHOW_PLOTS, verbose_output=VERBOSE_OUTPUT)

    dask_client = None
    # Always create a Dask client based on DASK_TYPE
    dask_client = get_dask_client(cluster_type=DASK_TYPE)
    if dask_client and hasattr(dask_client, 'dashboard_link'):
        logger.info(f"Dask client created: {dask_client.dashboard_link}")
    elif dask_client:
        logger.info("Dask client created (no dashboard link available for this cluster type).")

    try:
        if RUN_MODELING and processed_df is not None:
            lr_model, lr_metrics, rf_model, rf_metrics, xgb_model, xgb_metrics, stacked_model, stacked_metrics, X, y, preprocessor, stacked_input_example = run_model_pipeline(
                dask_client, processed_df, VERBOSE_OUTPUT, True, META_CLASSIFIER
            )

            # Perform final cross-validation for robustness
            lr_cv_mean, lr_cv_std = perform_final_model_evaluation(lr_model, X, y, preprocessor, "Logistic Regression", dask_client, VERBOSE_OUTPUT)
            rf_cv_mean, rf_cv_std = perform_final_model_evaluation(rf_model, X, y, preprocessor, "Random Forest", dask_client, VERBOSE_OUTPUT)
            xgb_cv_mean, xgb_cv_std = perform_final_model_evaluation(xgb_model, X, y, preprocessor, "XGBoost", dask_client, VERBOSE_OUTPUT)

            stacked_cv_mean, stacked_cv_std = None, None
            if True and stacked_model is not None:
                stacked_cv_mean, stacked_cv_std = perform_final_model_evaluation(stacked_model, X, y, preprocessor, "Stacked Ensemble", dask_client, VERBOSE_OUTPUT)

            log_analysis_results(start_time, DASK_TYPE, lr_metrics, rf_metrics, xgb_metrics, stacked_metrics, True, lr_cv_mean, lr_cv_std, rf_cv_mean, rf_cv_std, xgb_cv_mean, xgb_cv_std, stacked_cv_mean, stacked_cv_std)

            # MLflow Logging
            mlflow.log_metric("lr_cv_roc_auc", float(lr_cv_mean))
            mlflow.log_metric("rf_cv_roc_auc", float(rf_cv_mean))
            mlflow.log_metric("xgb_cv_roc_auc", float(xgb_cv_mean))
            if True and stacked_cv_mean is not None:
                mlflow.log_metric("stacked_cv_roc_auc", float(stacked_cv_mean))

            # Create a processed input example for MLflow logging
            # Ensure it's a Dask DataFrame head if X_processed_full is Dask
            if hasattr(X, 'head'):
                processed_input_example = preprocessor.transform(X.head(5))
            else:
                processed_input_example = preprocessor.transform(X[:5])

            # Log models
            mlflow.sklearn.log_model(lr_model, name="logistic_regression_model", input_example=processed_input_example)
            mlflow.sklearn.log_model(rf_model, name="random_forest_model", input_example=processed_input_example)
            mlflow.xgboost.log_model(xgb_model.named_steps['classifier'], name="xgboost_model", input_example=processed_input_example)
            if True and stacked_model is not None:
                mlflow.sklearn.log_model(stacked_model, name="stacked_ensemble_model", input_example=stacked_input_example[:5])
        
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
    finally:
        if dask_client:
            logger.info("Closing Dask client.")
            dask_client.close()
        logger.info(f"--- Heart Disease Analysis Finished in {time.time() - start_time:.2f} seconds ---")

if __name__ == "__main__":
    run_analysis()
        

