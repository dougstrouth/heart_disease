import unittest
import pandas as pd
import dask.dataframe as dd
from pandas.testing import assert_frame_equal
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
        self.assertEqual(processed_df['target'].sum(), 2)

        # Check if all columns from the schema are present
        for col in HeartDiseaseSchema.COLUMN_ORDER:
            self.assertIn(col, processed_df.columns)

        # Check data types
        for col, expected_type in HeartDiseaseSchema.COLUMNS.items():
            if col in processed_df.columns:
                self.assertTrue(pd.api.types.is_dtype_equal(processed_df[col].dtype, expected_type))

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
        self.assertEqual(processed_df['target'].sum(), 2)

        # Check if all columns from the schema are present
        for col in HeartDiseaseSchema.COLUMN_ORDER:
            self.assertIn(col, processed_df.columns)

        # Check data types
        for col, expected_type in HeartDiseaseSchema.COLUMNS.items():
            if col in processed_df.columns:
                self.assertTrue(pd.api.types.is_dtype_equal(processed_df[col].dtype, expected_type))


    @classmethod
    def tearDownClass(cls):
        """Close the Dask client."""
        cls.client.close()

if __name__ == '__main__':
    unittest.main()