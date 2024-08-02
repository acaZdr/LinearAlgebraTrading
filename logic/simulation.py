import numpy as np
from sklearn.decomposition import PCA


def perform_svd(df):
    # Select Columns for Analysis
    indicator_cols = ['SMA_10', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'BB_High', 'BB_Low', 'ATR_14']
    indicator_matrix = df[indicator_cols]

    # Standardize data
    indicator_matrix_std = (indicator_matrix - indicator_matrix.mean()) / indicator_matrix.std()

    # Perform SVD
    U, S, VT = np.linalg.svd(indicator_matrix_std, full_matrices=False)

    # Analyze singular values for explained variance
    print("Explained Variance Per Component:", S ** 2 / np.sum(S ** 2))

    return U, S, VT


def perform_pca(df, n_components=3):
    # Select Columns for Analysis
    indicator_cols = ['SMA_10', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 'BB_High', 'BB_Low', 'ATR_14']
    indicator_matrix = df[indicator_cols]

    # Standardize data
    indicator_matrix_std = (indicator_matrix - indicator_matrix.mean()) / indicator_matrix.std()

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(indicator_matrix_std)

    # Analyze PCA loadings to understand component composition
    print("PCA Loadings (Original Indicator Contributions):\n", pca.components_)

    return principal_components, pca
