import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from sklearn.decomposition import PCA
import numpy as np


class DataPreprocessor:
    def __init__(self, scaler_type='standard', use_pca=False):
        if scaler_type == 'standard':
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            self.scaler_volatility = StandardScaler()
            self.scaler_volume = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            self.scaler_volatility = MinMaxScaler()
            self.scaler_volume = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Use 'standard' or 'minmax'.")

        self.use_pca = use_pca
        self.pca = None
        self.quantile_33 = None
        self.quantile_67 = None

        self.fitted = False

    def fit_transform_data(self, X, y, volatility, volume, subfolder, n_components=0.95):
        if self.fitted:
            return self.transform_data(X, y, volatility, volume)

        X_scaled = self.scaler_X.fit_transform(X)
        torch.save(self.scaler_X, os.path.join(subfolder, 'scaler_X.pth'))

        self.quantile_33 = np.quantile(y, 0.33)
        self.quantile_67 = np.quantile(y, 0.67)

        y_quantized = np.digitize(y, [self.quantile_33, self.quantile_67])

        torch.save(self.scaler_y, os.path.join(subfolder, 'scaler_Y.pth'))

        volatility_scaled = self.scaler_volatility.fit_transform(volatility.values.reshape(-1, 1)).flatten()
        volume_scaled = self.scaler_volume.fit_transform(volume.values.reshape(-1, 1)).flatten()

        if self.use_pca:
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
            torch.save(self.pca, os.path.join(subfolder, "pca.pth"))

        self.fitted = True
        return X_scaled, y_quantized, volatility_scaled, volume_scaled

    def transform_data(self, X, y, volatility, volume):
        X_scaled = self.scaler_X.transform(X)
        y_quantized = np.digitize(y, [self.quantile_33, self.quantile_67])
        volatility_scaled = self.scaler_volatility.transform(volatility.values.reshape(-1, 1)).flatten()
        volume_scaled = self.scaler_volume.transform(volume.values.reshape(-1, 1)).flatten()
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return X_scaled, y_quantized, volatility_scaled, volume_scaled