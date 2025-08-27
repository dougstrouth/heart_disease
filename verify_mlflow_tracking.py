import mlflow
import random
from dotenv import load_dotenv

# Ensure MLflow is configured to use Databricks
# These environment variables should already be set from our previous steps
load_dotenv()
print("Starting MLflow verification run...")

# Set the MLflow experiment
mlflow.set_experiment("/Shared/Heart Disease Analysis")

# Start a new MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Log a parameter
    logged_param_value = "test_value_" + str(random.randint(1000, 9999))
    mlflow.log_param("verification_param", logged_param_value)
    print(f"Logged parameter 'verification_param': {logged_param_value}")

    # Log a metric
    logged_metric_value = random.uniform(0.0, 1.0)
    mlflow.log_metric("verification_metric", logged_metric_value)
    print(f"Logged metric 'verification_metric': {logged_metric_value}")

print("MLflow run completed. Now attempting to verify the run...")

# Search for the run to verify it was logged
try:
    # Search for runs that have the specific parameter we logged
    # Note: The filter string might need adjustment based on your MLflow version and backend
    # For Databricks, 'tags.mlflow.runName' is often the default run name if not specified
    # Or you can search by parameter: "params.verification_param = '{logged_param_value}'"
    
    # A more robust way to find the run is by its run_id
    found_runs = mlflow.search_runs(filter_string=f"attributes.run_id = '{run_id}'")

    if not found_runs.empty:
        print("\nVerification successful! Found the run in MLflow:")
        print(found_runs[['run_id', 'params.verification_param', 'metrics.verification_metric']])
        
        # Optional: Further assert values
        retrieved_param = found_runs['params.verification_param'].iloc[0]
        retrieved_metric = found_runs['metrics.verification_metric'].iloc[0]

        if retrieved_param == logged_param_value and abs(retrieved_metric - logged_metric_value) < 1e-6:
            print("Logged parameter and metric values match the retrieved values.")
        else:
            print("WARNING: Logged and retrieved values do NOT perfectly match. There might be a data type or precision issue.")
            print(f"Expected param: {logged_param_value}, Retrieved param: {retrieved_param}")
            print(f"Expected metric: {logged_metric_value}, Retrieved metric: {retrieved_metric}")

    else:
        print("\nVerification FAILED: Could not find the run in MLflow.")
        print("Please check your Databricks MLflow UI to confirm if the run appeared.")

except Exception as e:
    print(f"An error occurred during MLflow run search: {e}")

print("\nVerification script finished.")
