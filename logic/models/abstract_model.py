import abc
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib


class AbstractModel(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.pca = None
        self.model = None

    @abc.abstractmethod
    def build_model(self):
        """Build and return the actual model."""
        pass

    @abc.abstractmethod
    def preprocess_data(self, data, fit=False):
        """Preprocess the data, including feature engineering, scaling, and PCA."""
        pass

    @abc.abstractmethod
    def train(self, train_data, val_data):
        """Train the model using the provided training and validation data."""
        pass

    @abc.abstractmethod
    def predict(self, data):
        """Make predictions using the trained model."""
        pass

    @abc.abstractmethod
    def evaluate(self, test_data):
        """Evaluate the model's performance on test data."""
        pass

    def save_model(self, path):
        """Save the trained model to a file."""
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), path)
        elif isinstance(self.model, BaseEstimator):
            joblib.dump(self.model, path)
        else:
            raise NotImplementedError("Saving not implemented for this model type")

    def load_model(self, path):
        """Load a trained model from a file."""
        if isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(path))
        elif isinstance(self.model, BaseEstimator):
            import joblib
            self.model = joblib.load(path)
        else:
            raise NotImplementedError("Loading not implemented for this model type")

    @staticmethod
    def choose_n_components(X_scaled, variance_threshold=0.95):
        """Choose the number of components for PCA."""
        pca = PCA().fit(X_scaled)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(cumulative_variance_ratio >= variance_threshold)[0][0] + 1
        return n_components
    