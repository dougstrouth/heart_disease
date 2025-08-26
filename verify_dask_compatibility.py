import dask.dataframe as dd
from dask_data_utils import perform_eda
from data_schema import HeartDiseaseSchema
from utils.logger_config import setup_logging
from config import DASK_TYPE, GCS_DATA_PATH

# Setup logging
logger = setup_logging()

# Define dtype specification
dtype_spec = {'ca': 'object', 'cp': 'object', 'restecg': 'object', 'sex': 'object', 'slope': 'object', 'thal': 'object'}

if DASK_TYPE in ['coiled', 'cloud']:
    logger.info(f"Loading data from GCS: {GCS_DATA_PATH}")
    ddf = dd.read_csv(GCS_DATA_PATH, dtype=dtype_spec)
else:
    logger.info("Loading data from local CSV: combined_heart_disease_dataset.csv")
    ddf = dd.read_csv('combined_heart_disease_dataset.csv', dtype=dtype_spec)


# Print the head to verify it's loaded correctly
logger.info("Head of the Dask DataFrame:")
logger.info(ddf.head())

# Perform EDA using the function from dask_data_utils
logger.info("\nPerforming EDA on the Dask DataFrame...")
perform_eda(
    ddf,
    "Combined Dataset (Dask)",
    numerical_features=HeartDiseaseSchema.NUMERICAL_COLUMNS(),
    categorical_features=HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE(),
    show_plots=False,
    verbose_output=True
)

logger.info("\nDask compatibility check passed!")