from sklearn.linear_model import LogisticRegression

# --- Configuration ---
SHOW_PLOTS = False  # Set to True to display plots, False to suppress them
VERBOSE_OUTPUT = False # Set to True for more detailed print statements
# Dask type can be 'local', 'coiled', or 'cloud'. 'cloud' uses a local Dask client but loads data from GCS.
DASK_TYPE = 'cloud'

# --- Configuration for Automated Parameter Search ---
TARGET_RUN_TIME_MINUTES = 5.0  # Target maximum runtime for the full analysis
MAX_SEARCH_TIME_MINUTES = 5.0 # Maximum time to spend searching for optimal parameters
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

CV_FOLDS = 5 # Number of cross-validation folds for GridSearchCV/RandomizedSearchCV

# --- Configuration for Stacked Ensemble ---
RUN_STACKED_ENSEMBLE = True # Set to True to run the stacked ensemble
# Define the meta-classifier for the stacked ensemble
META_CLASSIFIER = LogisticRegression(solver='liblinear', random_state=42)