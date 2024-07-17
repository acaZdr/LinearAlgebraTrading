import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import svd

from data_preprocessing import import_data

# Load your data
data = import_data('data/BTCUSDT-1s-2023-11.csv', limit=10000)

# Ensure no NaN values in the data
data = dropna(data)

# Compute technical indicators
data = add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume")

# Select the top 10 technical indicators for demonstration purposes
# Here, we select a variety of indicators manually
original_data = ['Open', 'High', 'Low', 'Close', 'Volume']
indicators = ['trend_sma_fast', 'trend_ema_fast', 'momentum_rsi',  # 'momentum_stoch_k',
              'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volume_adi',
              'volume_obv', 'volume_cmf']
needed_columns = original_data + indicators

# Prepare the data matrix with the selected indicators
indicator_data = data[needed_columns]

indicator_data = dropna(indicator_data)

# Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(indicator_data)

# Perform SVD
U, S, Vt = svd(standardized_data)

n = indicator_data.shape[1]

pca = PCA(n_components=n)
pca_data = pca.fit_transform(standardized_data)

cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

print(f"Number of Components: {n}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative Explained Variance Ratio: {cumulative_explained_variance_ratio}")


# n_components = n
# pca = PCA(n_components=n_components)
# pca_data = pca.fit_transform(standardized_data)
#
# # Create a DataFrame with the transformed data of the first 4 components
# pca_df = pd.DataFrame(data=pca_data[:, :n_components], columns=[f"Component_{i + 1}" for i in range(n_components)])

# Save the DataFrame to a CSV file
# pca_df.to_csv('pca_transformed_data.csv', index=False)