# --- Orchestration ---
RUN_ANALYSIS = False
HYPERPARAMETER_TUNING_CONFIG = {
    "run": True,
    "cloud_run_mode": "full",  # "test" or "full"
    "coiled_max_time_hours": 1.1,
}
RUN_EXPERIMENT_REVIEW = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Configuration ---
SHOW_PLOTS = False  # Set to True to display plots, False to suppress them
VERBOSE_OUTPUT = False # Set to True for more detailed print statements
# Dask type can be 'local', 'coiled', or 'cloud'. 'cloud' uses a local Dask client but loads data from GCS.
DASK_TYPE = 'coiled'

# --- GCS Configuration ---
GCS_DATA_PATH = "gs://my-heart-disease-data-bucket/data/combined_heart_disease_dataset_3.csv"

# --- Local Data Paths ---
LOCAL_UCI_PATH = 'tests/test_data/dummy_uci_data.csv'
LOCAL_SYNTHETIC_PATH = 'tests/test_data/dummy_synthetic_data.csv'
LOCAL_JOHNSMITH_PATH = 'tests/test_data/dummy_johnsmith_data.csv'

# --- Configuration for Automated Parameter Search ---
TARGET_RUN_TIME_MINUTES = 5.0  # Target maximum runtime for the full analysis
MAX_SEARCH_TIME_MINUTES = 10.0 # Maximum time to spend searching for optimal parameters
RUN_PARAMETER_SEARCH = True   # Set to True to run the automated parameter search

# For local testing with limited resources, consider reducing the search space:
# - Decrease the number of options in LR_C_OPTIONS, RF_N_ESTIMATORS_OPTIONS, etc.
# - Reduce RF_RANDOM_SEARCH_N_ITER.
# - Set RUN_PARAMETER_SEARCH to False to use minimal grids.
# - Reduce the 'cv' (cross-validation) folds in train_evaluate_model (currently hardcoded to 10).

# Define parameter options to iterate through for automated search
LR_C_OPTIONS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 20000.0, 50000.0]
RF_N_ESTIMATORS_OPTIONS = [100, 200, 300]
RF_MAX_DEPTH_OPTIONS = [5, 10, 15]
RF_MIN_SAMPLES_SPLIT_OPTIONS = [2]
RF_MIN_SAMPLES_LEAF_OPTIONS = [1]
XGB_N_ESTIMATORS_OPTIONS = [100, 200, 300]
XGB_LEARNING_RATE_OPTIONS = [0.01, 0.1, 0.2]


RF_RANDOM_SEARCH_N_ITER = 20 # Number of iterations for RandomizedSearchCV for Random Forest
LR_RANDOM_SEARCH_N_ITER = 20 # Number of iterations for RandomizedSearchCV for Logistic Regression
XGB_RANDOM_SEARCH_N_ITER = 20 # Number of iterations for RandomizedSearchCV for XGBoost

CV_FOLDS = 7 # Number of cross-validation folds for GridSearchCV/RandomizedSearchCV

# --- Coiled-Specific Configuration for Enhanced Search (Optional) ---
# These parameters will be used when DASK_TYPE is 'coiled' or 'cloud'
# to allow for more extensive hyperparameter tuning.

COILED_LR_C_OPTIONS = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 20000.0, 50000.0, 100000.0]
COILED_RF_N_ESTIMATORS_OPTIONS = [100, 200, 300, 500, 700]
COILED_RF_MAX_DEPTH_OPTIONS = [5, 10, 15, 20, 25, None] # None for unlimited depth
COILED_RF_MIN_SAMPLES_SPLIT_OPTIONS = [2, 5, 10]
COILED_RF_MIN_SAMPLES_LEAF_OPTIONS = [1, 2, 4]
COILED_XGB_N_ESTIMATORS_OPTIONS = [100, 200, 300, 500, 700]
COILED_XGB_LEARNING_RATE_OPTIONS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]

COILED_RF_RANDOM_SEARCH_N_ITER = 50 # More iterations for Random Forest
COILED_LR_RANDOM_SEARCH_N_ITER = 50 # More iterations for Logistic Regression
COILED_XGB_RANDOM_SEARCH_N_ITER = 50 # More iterations for XGBoost

# --- Configuration for Stacked Ensemble ---
RUN_STACKED_ENSEMBLE = True # Set to True to run the stacked ensemble
RUN_MODELING = True # Set to True to run the modeling pipeline
# Define the meta-classifier for the stacked ensemble
META_CLASSIFIER = LogisticRegression(solver='liblinear', random_state=42)

# --- Coiled-Specific Meta-Classifier Options (Optional) ---
# Define a list of meta-classifiers to try for stacking
COILED_META_CLASSIFIERS = [
    LogisticRegression(solver='liblinear', random_state=42),
    RandomForestClassifier(random_state=42, n_estimators=50),
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=50)
]