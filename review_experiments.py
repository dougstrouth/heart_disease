import mlflow

def review_and_register_model(experiment_name="/Shared/Heart Disease Analysis", top_n=1):
    """
    Reviews the runs in an experiment, finds the best one, and registers the model.
    """
    # Search for runs in the experiment
    runs = mlflow.search_runs(experiment_names=[experiment_name])

    # Sort by roc_auc and get the best run
    best_run = runs.sort_values("metrics.roc_auc", ascending=False).iloc[0]

    print("Best Run:")
    print(f"  Run ID: {best_run.run_id}")
    print(f"  ROC AUC: {best_run['metrics.roc_auc']:.4f}")
    print("  Parameters:")
    for key, value in best_run.filter(regex='params').to_dict().items():
        print(f"    {key.replace('params.', '')}: {value}")


    # Register the model
    model_uri = f"runs:/{best_run.run_id}/random_forest_model"
    model_name = "HeartDiseaseRandomForest"
    
    print(f"\nRegistering model {model_name} from run {best_run.run_id}")
    
    registered_model = mlflow.register_model(model_uri, model_name)
    
    print(f"Model registered as version {registered_model.version}")

if __name__ == "__main__":
    review_and_register_model()
