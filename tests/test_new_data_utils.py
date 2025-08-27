import unittest
import pandas as pd
from dask.distributed import Client

from pandas_data_utils import get_processed_data as pandas_get_processed_data
from dask_data_utils import get_processed_data as dask_get_processed_data
from data_schema import HeartDiseaseSchema

class TestPandasDataUtils(unittest.TestCase):

    def test_get_processed_data(self):
        """Test the entire pandas data processing pipeline."""
        processed_df = pandas_get_processed_data(
            uci_path='tests/test_data/dummy_uci_data.csv',
            synthetic_path='tests/test_data/dummy_synthetic_data.csv',
            johnsmith_path='tests/test_data/dummy_johnsmith_data.csv'
        )

        self.assertEqual(processed_df.shape[0], 3)
        self.assertIn('origin', processed_df.columns)
        self.assertEqual(processed_df[HeartDiseaseSchema.TARGET_COLUMN].sum(), 2)

        # Check if all columns from the schema are present
        for col in HeartDiseaseSchema.COLUMN_ORDER:
            self.assertIn(col, processed_df.columns)

        # Check data types
        for col, expected_type in HeartDiseaseSchema.COLUMNS.items():
            if col in processed_df.columns:
                if expected_type is int:
                    self.assertTrue(pd.api.types.is_integer_dtype(processed_df[col].dtype), f"Column {col} expected integer, got {processed_df[col].dtype}")
                elif expected_type is float:
                    self.assertTrue(pd.api.types.is_float_dtype(processed_df[col].dtype), f"Column {col} expected float, got {processed_df[col].dtype}")
                else:
                    # For other types (like object for categorical), allow 'object' or string dtypes
                    if expected_type is object:
                        self.assertTrue(
                            processed_df[col].dtype == object or pd.api.types.is_string_dtype(processed_df[col].dtype),
                            f"Column {col} expected object or string dtype, got {processed_df[col].dtype}"
                        )
                    else:
                        self.assertTrue(pd.api.types.is_dtype_equal(processed_df[col].dtype, expected_type), f"Column {col} expected {expected_type}, got {processed_df[col].dtype}")

class TestDaskDataUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a Dask client for the tests."""
        cls.client = Client()

    def test_get_processed_data(self):
        """Test the entire dask data processing pipeline."""
        processed_df = dask_get_processed_data(
            uci_path='tests/test_data/dummy_uci_data.csv',
            synthetic_path='tests/test_data/dummy_synthetic_data.csv',
            johnsmith_path='tests/test_data/dummy_johnsmith_data.csv'
        ).compute()

        self.assertEqual(processed_df.shape[0], 3)
        self.assertIn('origin', processed_df.columns)
        self.assertEqual(processed_df[HeartDiseaseSchema.TARGET_COLUMN].sum(), 2)

        # Check if all columns from the schema are present
        for col in HeartDiseaseSchema.COLUMN_ORDER:
            self.assertIn(col, processed_df.columns)

        # Check data types
        for col, expected_type in HeartDiseaseSchema.COLUMNS.items():
            if col in processed_df.columns:
                if expected_type is int:
                    self.assertTrue(pd.api.types.is_integer_dtype(processed_df[col].dtype), f"Column {col} expected integer, got {processed_df[col].dtype}")
                elif expected_type is float:
                    self.assertTrue(pd.api.types.is_float_dtype(processed_df[col].dtype), f"Column {col} expected float, got {processed_df[col].dtype}")
                else:
                    # For other types (like object for categorical), allow 'object' or string dtypes
                    if expected_type is object:
                        self.assertTrue(
                            processed_df[col].dtype == object or pd.api.types.is_string_dtype(processed_df[col].dtype),
                            f"Column {col} expected object or string dtype, got {processed_df[col].dtype}"
                        )
                    else:
                        self.assertTrue(pd.api.types.is_dtype_equal(processed_df[col].dtype, expected_type), f"Column {col} expected {expected_type}, got {processed_df[col].dtype}")


    @classmethod
    def tearDownClass(cls):
        """Close the Dask client."""
        cls.client.close()

if __name__ == '__main__':
    unittest.main()