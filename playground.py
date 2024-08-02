import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import argparse
import yaml
import logging

from data_preprocessing import import_data


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def preprocess_data(data, config):
    """Preprocess the data by computing technical indicators and selecting features."""
    data = dropna(data)
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume")

    original_data = config['original_columns']
    indicators = config['indicator_columns']
    needed_columns = original_data + indicators

    indicator_data = data[needed_columns]
    return dropna(indicator_data)


def standardize_data(data):
    """Standardize the data using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def perform_pca(data, n_components):
    """Perform PCA on the data."""
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca, pca_data


def plot_explained_variance(pca):
    """Plot the explained variance ratio."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.savefig('explained_variance.png')
    plt.close()


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load data
        data = import_data(config['data_path'], 10000)
        logging.info(f"Data loaded successfully. Shape: {data.shape}")

        # Preprocess data
        processed_data = preprocess_data(data, config)
        logging.info(f"Data preprocessed. Shape: {processed_data.shape}")

        # Standardize data
        standardized_data = standardize_data(processed_data)
        logging.info("Data standardized.")

        # Perform PCA
        n_components = processed_data.shape[1]
        pca, pca_data = perform_pca(standardized_data, n_components)
        logging.info("PCA performed.")

        # Print results
        print(f"Number of Components: {n_components}")
        print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative Explained Variance Ratio: {np.cumsum(pca.explained_variance_ratio_)}")

        # Plot explained variance
        plot_explained_variance(pca)
        logging.info("Explained variance plot saved as 'explained_variance.png'")

        # Save PCA results
        pca_df = pd.DataFrame(data=pca_data, columns=[f"Component_{i + 1}" for i in range(n_components)])
        pca_df.to_csv('pca_transformed_data.csv', index=False)
        logging.info("PCA transformed data saved to 'pca_transformed_data.csv'")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform PCA on cryptocurrency data.")
    parser.add_argument('--config', default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)
