import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import argparse
import logging
import traceback
from src.data_preprocessing.data_importer import import_data
from ta import add_all_ta_features
import matplotlib.pyplot as plt



def choose_n_components(X_scaled, variance_threshold=0.95):
    pca = PCA().fit(X_scaled)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumulative_variance_ratio >= variance_threshold)[0][0] + 1
    return n_components


def preprocess_data(data, config):
    """Preprocess the data by calculating technical indicators, selecting features, and applying PCA."""
    # Calculate technical indicators
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume")

    # Drop OHLCV columns from the dataset
    data = data.drop(columns=['Open', 'High', 'Low', 'Volume'])

    # Select the target column
    target = config['target']

    # Ensure the target column is present
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is missing in the dataset")

    # Replace infinite values with NaNs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns with NaN values
    data = data.dropna(axis=1, how='any')

    # Check if there is any data left after dropping columns
    if data.empty or data.shape[1] <= 1:
        raise ValueError("No data available after dropping columns with NaN values")

    # Split combined data back into features and target
    X  = data.drop(columns=[target])
    y = data[target].values

    # Convert features to numeric, replacing non-numeric values with NaN
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop columns with all NaN values
    X = X.dropna(axis=1, how='all')

    # Check for any remaining NaN values in features
    if X.isnull().values.any():
        raise ValueError("Input X contains NaN values after converting to numeric")

    # Convert to numpy array
    X = X.values

    # Standardize the features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Perform PCA on all features
    n_components = choose_n_components(X_scaled, variance_threshold=0.95)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Scale the target
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Ensure y_scaled is a 1D array

    return X_pca, y_scaled, pca, scaler_X, scaler_y


def train_and_evaluate_linear_regression(train_data, val_data, config):
    X_train, y_train, pca, scaler_X, scaler_y = preprocess_data(train_data, config)
    X_val, y_val, _, _, _ = preprocess_data(val_data, config)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = model.predict(X_val)
    val_loss = mean_squared_error(y_val, y_pred)

    # Inverse transform the predictions to the original scale
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    return val_loss, y_val_orig, y_pred_orig


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load training data
        train_data = import_data(config['train_data_path'], limit=config.get('data_limit'))
        logging.info(f"Training data loaded successfully. Shape: {train_data.shape}")

        # Load validation data
        val_data = import_data(config['val_data_path'], limit=config.get('data_limit'))
        logging.info(f"Validation data loaded successfully. Shape: {val_data.shape}")

        # Train and evaluate linear regression model
        val_loss, y_val_orig, y_pred_orig = train_and_evaluate_linear_regression(train_data, val_data, config)

        logging.info(f"Validation Loss (MSE) for Linear Regression: {val_loss:.4f}")

        # Optionally, plot the actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_val_orig, label='Actual')
        plt.plot(y_pred_orig, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Bitcoin Price')
        plt.title('Actual vs Predicted Bitcoin Price')
        plt.legend()
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(f"Error traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a simple linear regression model for cryptocurrency trading.")
    parser.add_argument('--config', default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)
