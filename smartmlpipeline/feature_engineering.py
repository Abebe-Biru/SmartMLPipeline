# smartmlpipeline/feature_engineering.py

from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        self.selector = None

    def select_best_features(self, X, y, k='all'):
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_new = self.selector.fit_transform(X, y)
        return pd.DataFrame(X_new)

    def get_support_indices(self):
        return self.selector.get_support(indices=True)
