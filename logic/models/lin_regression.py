import logging
import os
import traceback

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ta import add_all_ta_features

from logic.models.lstm import subfolder, project_root
from src.data_preprocessing.data_importer import import_data
from src.utils.config_loader import load_config
from src.utils.data_saving_and_displaying import save_and_display_results

from abstract_model import AbstractModel


class LinearRegressionModel(AbstractModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        self.model = LinearRegression()

    def preprocess_data(self, data, fit=False):
        # Shift the target variable
        target = self.config['target']
        data['target'] = data[target].shift(-1)  # Shift the target to predict next minute's close

        # Calculate technical indicators
        data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)

        # Drop the last row as it won't have a target value
        data = data.dropna().reset_index(drop=True)

        look_ahead_indicators = ['trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a',
                                 'trend_visual_ichimoku_b', 'trend_stc', 'trend_psar_up', 'trend_psar_down']

        # Drop OHLCV columns from the dataset, keeping only the indicators and target
        feature_columns = [col for col in data.columns if col not in
                           (['date', 'Open', 'High', 'Low', 'Volume', 'target'] + look_ahead_indicators)]

        # Ensure all feature columns exist in the dataframe
        feature_columns = [col for col in feature_columns if col in data.columns]

        X = data[feature_columns].values
        y = data['target'].values

        if fit:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

            if self.config.get('use_pca', False):
                n_components = self.choose_n_components(X_scaled,
                                                        variance_threshold=self.config.get('variance_threshold', 0.95))
                self.pca = PCA(n_components=n_components)
                X_scaled = self.pca.fit_transform(X_scaled)
        else:
            X_scaled = self.scaler_X.transform(X)
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()

            if self.pca is not None:
                X_scaled = self.pca.transform(X_scaled)

        return X_scaled, y_scaled

    def train(self, train_data, val_data):
        X_train, y_train = self.preprocess_data(train_data, fit=True)
        X_val, y_val = self.preprocess_data(val_data, fit=False)

        self.model.fit(X_train, y_train)

        train_predictions = self.model.predict(X_train)
        val_predictions = self.model.predict(X_val)

        train_loss = mean_squared_error(y_train, train_predictions)
        val_loss = mean_squared_error(y_val, val_predictions)

        logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return train_loss, val_loss

    def predict(self, data):
        X, _ = self.preprocess_data(data, fit=False)
        return self.model.predict(X)

    def evaluate(self, test_data):
        X_test, y_test = self.preprocess_data(test_data, fit=False)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse


def main(config_path):
    config = load_config(config_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        logging.info("Starting main function")
        logging.info(f"Configuration loaded from: {config_path}")

        # Load and process datasets
        train_data = import_data(os.path.join(project_root, 'data', config['train_data']),
                                 limit=config['data_limit'])
        val_data = import_data(os.path.join(project_root, 'data', config['val_data']),
                               limit=config['data_limit'])
        test_data = import_data(os.path.join(project_root, 'data', config['test_data']),
                                limit=config['data_limit'])

        # Initialize and train the model
        model = LinearRegressionModel(config)
        model.build_model()
        train_loss, val_loss = model.train(train_data, val_data)

        logging.info(f"Training completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Evaluate on the test data
        test_loss = model.evaluate(test_data)
        logging.info(f"Test Loss: {test_loss:.4f}")

        # Make predictions on validation data
        val_predictions = model.predict(val_data)

        # Inverse transform the predictions and actual values
        val_actual = model.scaler_y.inverse_transform(
            model.preprocess_data(val_data, fit=False)[1].reshape(-1, 1)).flatten()
        val_predictions = model.scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()

        # Save and display results
        save_and_display_results(val_actual, val_predictions, subfolder)
        logging.info(f"Results saved in {subfolder}")

        # Save the model
        model.save_model(os.path.join(subfolder, 'linear_regression_model.joblib'))

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc()

    logging.info("Main completed")


if __name__ == '__main__':
    config_path = '../../config/config.yaml'
    main(config_path)
