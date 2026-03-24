import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class OutliersDetector(BaseEstimator, TransformerMixin):
    def __init__(self, random_state):
        self.random_state=random_state
        self.model = None

    def fit(self, X, y=None):
        self.model = IsolationForest(random_state=self.random_state)
        self.model.fit(X)
        self._is_fitted = True 
        return self
    
    def transform(self, X):
        return np.concatenate([X, np.array([self.model.predict(X), self.model.score_samples(X)]).T], axis=1)
    
    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted