from typing import Dict, List, ClassVar

class HeartDiseaseSchema:
    """
    Defines the schema for the heart disease dataset, including column names,
    data types, and categorical mappings.
    """

    # Define expected column names and their data types
    COLUMNS: Dict[str, type] = {
        "age": int,
        "sex": object, # Will be mapped to categorical later
        "cp": object,  # Will be mapped to categorical later
        "trestbps": int,
        "chol": int,
        "fbs": object, # Will be mapped to categorical later
        "restecg": object, # Will be mapped to categorical later
        "thalach": int,
        "exang": object, # Will be mapped to categorical later
        "oldpeak": float,
        "slope": object, # Will be mapped to categorical later
        "ca": object,  # Will be mapped to categorical later
        "thal": object, # Will be mapped to categorical later
        "smoking": object, # Will be mapped to categorical later
        "diabetes": object, # Will be mapped to categorical later
        "bmi": float,
        "heart_disease": int, # Target variable
    }

    # Define categorical mappings for columns that are encoded as integers
    CATEGORICAL_MAPPINGS: Dict[str, Dict[int, str]] = {
        "sex": {0: "female", 1: "male"},
        "cp": {
            0: "typical angina",
            1: "atypical angina",
            2: "non-anginal pain",
            3: "asymptomatic",
        },
        "fbs": {0: "false", 1: "true"}, # >120 mg/dl
        "restecg": {
            0: "normal",
            1: "ST-T wave abnormality",
            2: "left ventricular hypertrophy",
        },
        "exang": {0: "no", 1: "yes"},
        "slope": {
            0: "upsloping",
            1: "flat",
            2: "downsloping"
        },
        "ca": {0: "0", 1: "1", 2: "2", 3: "3"}, # Number of major vessels
        "thal": {
            0: "normal",
            1: "fixed defect",
            2: "reversible defect",
            3: "normal", # Some datasets might use 3 for normal
        },
        "smoking": {0: "no", 1: "yes"},
        "diabetes": {0: "no", 1: "yes"},
    }

    # Define the target column
    TARGET_COLUMN: str = "heart_disease"

    # Define expected ranges for numerical columns (optional, for more strict validation)
    NUMERICAL_RANGES: Dict[str, tuple] = {
        "age": (0, 120), # Reasonable age range
        "trestbps": (80, 200), # Typical blood pressure range
        "chol": (100, 600), # Typical cholesterol range
        "thalach": (60, 220), # Typical max heart rate range
        "oldpeak": (0.0, 6.0), # ST depression
        "bmi": (10.0, 60.0), # Typical BMI range
    }

    COLUMN_ORDER: ClassVar[List[str]] = list(COLUMNS.keys())

    @staticmethod
    def CATEGORICAL_COLUMNS_TO_ENCODE() -> List[str]:
        return list(HeartDiseaseSchema.CATEGORICAL_MAPPINGS.keys())

    @staticmethod
    def NUMERICAL_COLUMNS() -> List[str]:
        return [
            col for col, dtype in HeartDiseaseSchema.COLUMNS.items() if dtype in [int, float] and col not in HeartDiseaseSchema.CATEGORICAL_COLUMNS_TO_ENCODE() and col != HeartDiseaseSchema.TARGET_COLUMN
        ]