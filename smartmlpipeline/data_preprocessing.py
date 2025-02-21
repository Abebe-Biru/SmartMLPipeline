# smartmlpipeline/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.imputers = {}
        self.encoders = {}
        self.scaler = StandardScaler()

    def handle_missing_values(self, df, strategy='mean'):
        for column in df.columns:
            if df[column].isnull().any():
                imputer = SimpleImputer(strategy=strategy)
                df[[column]] = imputer.fit_transform(df[[column]])
                self.imputers[column] = imputer
        return df

    def encode_categorical(self, df, categorical_columns):
        for column in categorical_columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            self.encoders[column] = encoder
        return df

    def scale_features(self, df, numerical_columns):
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
