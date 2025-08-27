from config import RUN_ANALYSIS, HYPERPARAMETER_TUNING_CONFIG, RUN_EXPERIMENT_REVIEW
from heart_disease_analysis import run_analysis
from hyperparameter_tuning import run_tuning
from review_experiments import review_and_register_model
from utils.logger_config import setup_logging

def main():
    """
    Main function to orchestrate the different parts of the project.
    """
    logger = setup_logging()
    logger.info("--- Starting Orchestration ---")

    if RUN_ANALYSIS:
        logger.info("--- Running Full Analysis ---")
        run_analysis()

    if HYPERPARAMETER_TUNING_CONFIG.get("run", False):
        logger.info("--- Running Hyperparameter Tuning ---")
        run_tuning(HYPERPARAMETER_TUNING_CONFIG)

    if RUN_EXPERIMENT_REVIEW:
        logger.info("--- Running Experiment Review ---")
        review_and_register_model()

    logger.info("--- Orchestration Finished ---")

if __name__ == "__main__":
    main()
