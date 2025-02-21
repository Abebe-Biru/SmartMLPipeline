# tests/test_data_preprocessing.py

import unittest
import pandas as pd
from smartmlpipeline.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.df = pd.DataFrame({
            'numerical': [1, 2, None],
            'categorical': ['a', 'b', None]
        })

    def test_handle_missing_values(self):
        df_clean = self.preprocessor.handle_missing_values(self.df)
        self.assertFalse(df_clean.isnull().values.any())

    def test_encode_categorical(self):
        df_filled = self.df.fillna('missing')
        df_encoded = self.preprocessor.encode_categorical(df_filled, ['categorical'])
        self.assertTrue(df_encoded['categorical'].dtype in [int, 'int32', 'int64'])

    def test_scale_features(self):
        df_filled = self.df.fillna(0)
        df_scaled = self.preprocessor.scale_features(df_filled, ['numerical'])
        self.assertAlmostEqual(df_scaled['numerical'].mean(), 0, places=5)

if __name__ == '__main__':
    unittest.main()
