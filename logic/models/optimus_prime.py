import logging
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from ta import add_all_ta_features

from src.data_preprocessing.data_importer import import_data
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.utils.config_loader import load_config
from src.utils.data_saving_and_displaying import save_and_display_results_classification


def setup_logging(subfolder):
    log_filename = os.path.join(subfolder, f'experiment_log_{time.strftime("%Y%m%d-%H%M")}.log')

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def calculate_volatility(data, window_size=20):
    data['log_return'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
    data['volatility'] = data['log_return'].rolling(window=window_size).std()
    return data['volatility'].fillna(method='bfill')


def preprocess_data(data, config, data_preprocessor, subfolder):
    data = data.copy()
    data['Close_diff'] = data['Close'].pct_change()
    data['target'] = data[config['target']].shift(-1)
    data = data.dropna().reset_index(drop=True)

    absolute_prices = data['Close'].values

    data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
    data = data.dropna().reset_index(drop=True)

    look_ahead_indicators = ['trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a',
                             'trend_visual_ichimoku_b', 'trend_stc', 'trend_psar_up', 'trend_psar_down']

    feature_columns = [col for col in data.columns if col not in (
            ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'Average_Close_diff'] + look_ahead_indicators)]

    data['volatility'] = calculate_volatility(data, window_size=config.get('volatility_window_size', 20))
    data = data.drop(columns=['Close'])
    data = data.dropna().reset_index(drop=True)

    volatility = data['volatility']
    volume = data['Volume']

    X = data[feature_columns]
    y = data['target']

    X_scaled, y_scaled, volatility_scaled, volume_scaled = data_preprocessor.fit_transform_data(X, y, volatility, volume, subfolder)

    # Ensure y_scaled is a NumPy array
    if isinstance(y_scaled, list):
        y_scaled = np.array(y_scaled)

    # Check if the number of columns in X_scaled matches the number of feature columns
    if X_scaled.shape[1] != len(feature_columns):
        logging.warning(f"Number of columns in scaled data ({X_scaled.shape[1]}) does not match number of feature columns ({len(feature_columns)})")
        logging.warning("Adjusting feature columns to match scaled data")
        feature_columns = [f'feature_{i}' for i in range(X_scaled.shape[1])]

    # Create a new DataFrame with scaled data
    processed_data = pd.DataFrame(X_scaled, columns=feature_columns)
    processed_data['target'] = y_scaled
    processed_data['volatility'] = volatility_scaled
    processed_data['volume'] = volume_scaled
    processed_data['date'] = data['date']
    processed_data['group_id'] = 'crypto'  # Assuming a single group for all data

    return processed_data, absolute_prices

class CryptoTransformer(pl.LightningModule):
    def __init__(self, config, training_data):
        super().__init__()
        self.config = config
        self.training_data = training_data

        loss = CrossEntropy()
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_data,
            learning_rate=self.config['learning_rate'],
            hidden_size=self.config['hidden_dim'],
            attention_head_size=self.config['attention_head_size'],
            dropout=self.config['dropout'],
            hidden_continuous_size=self.config['hidden_continuous_size'],
            output_size=self.config['output_size'],
            loss=loss,
            logging_metrics=None,
        )

        # Ignore 'training_data' and 'model' along with others to avoid warnings
        self.save_hyperparameters(ignore=['training_data', 'model', 'loss', 'logging_metrics'])
        self.model = self.model.to(self.device)

    def forward(self, x):
        logging.info(f"Forward input type: {type(x)}")
        if isinstance(x, dict):
            x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        elif isinstance(x, tuple):
            logging.info(f"Forward tuple structure: {[type(item) for item in x]}")
            x = tuple(item.to(self.device) if isinstance(item, torch.Tensor) else item for item in x)
        else:
            x = x.to(self.device)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        logging.info(f"Batch type: {type(batch)}")
        logging.info(f"Batch structure: {[type(item) for item in batch]}")
        x, y = batch
        logging.info(f"x type: {type(x)}, y type: {type(y)}")

        # Move x to the correct device
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        elif isinstance(x, dict):
            x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

        # Handle y being a list or other non-tensor types
        if isinstance(y, torch.Tensor):
            y = y.to(self.device)
        elif isinstance(y, list):
            y = torch.tensor(y, device=self.device, dtype=torch.float32)  # Adjust dtype as needed
        elif isinstance(y, tuple):
            y = tuple(item.to(self.device) if isinstance(item, torch.Tensor) else item for item in y)
        else:
            logging.warning(f"Unsupported type for y: {type(y)}")
            raise TypeError(f"Unsupported type for y: {type(y)}")

        outputs = self(x)
        loss = self.model.loss(outputs, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Ensure y is already a tensor; remove unnecessary conversion to torch.tensor
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=torch.float32)

        # If y has multiple elements, ensure correct dimensionality
        y = y.to(self.device, dtype=torch.float32)

        # Handle multi-dimensional y if needed
        if y.dim() > 1:
            y = y.squeeze()  # Removes extra dimensions, like [batch_size, 1] to [batch_size]

        # Forward pass
        output = self(x)

        # Compute loss
        loss = self.loss_function(output, y)

        # Log validation loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])

def main(config_path):
    config = load_config(config_path)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subfolder = os.path.join(project_root, 'results', f'experiment_{time.strftime("%Y%m%d-%H%M")}')
    os.makedirs(subfolder, exist_ok=True)

    setup_logging(subfolder)

    try:
        logging.info("Starting main function")
        logging.info(f"Configuration loaded from: {config_path}")

        all_data_filenames = config['train_data'] + config['val_data'] + config['test_data']
        all_data_paths = [os.path.join(project_root, 'data', path) for path in all_data_filenames]
        all_data = import_data(all_data_paths, limit=config.get('data_limit', None))
        all_data = all_data.sort_values('date').reset_index(drop=True)

        test_split = int(0.8 * len(all_data))
        train_val_data = all_data[:test_split]
        test_data = all_data[test_split:]

        val_split = int(0.8 * len(train_val_data))
        train_data = train_val_data[:val_split]
        val_data = train_val_data[val_split:]

        logging.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

        data_preprocessor = DataPreprocessor(scaler_type=config.get('scaler', 'standard'),
                                             use_pca=config.get('use_pca', False))

        train_processed, train_prices = preprocess_data(train_data, config, data_preprocessor, subfolder)
        val_processed, val_prices = preprocess_data(val_data, config, data_preprocessor, subfolder)
        test_processed, test_prices = preprocess_data(test_data, config, data_preprocessor, subfolder)

        max_encoder_length = config['max_encoder_length']
        max_prediction_length = config['max_prediction_length']

        training_cutoff = train_processed['date'].max() - pd.Timedelta(days=max_prediction_length)

        # Convert 'date' column to Unix timestamp in seconds (or milliseconds)
        train_processed['time_idx'] = np.arange(len(train_processed))
        val_processed['time_idx'] = np.arange(len(val_processed))

        training_data = TimeSeriesDataSet(
            train_processed,
            time_idx='time_idx',
            target='target',
            group_ids=['group_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=['group_id'],
            time_varying_known_reals=['date'],
            time_varying_unknown_reals=[col for col in train_processed.columns if
                                        col not in ['date', 'group_id', 'target']],
            target_normalizer=None,  # No normalization for classification
        )

        validation_data = TimeSeriesDataSet.from_dataset(training_data, val_processed, predict=True,
                                                         stop_randomization=True)

        train_dataloader = training_data.to_dataloader(
            train=True,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        val_dataloader = validation_data.to_dataloader(
            train=False,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        # Initialize model
        model = CryptoTransformer(config, training_data)

        # Training
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        trainer = pl.Trainer(
            max_epochs=config['num_epochs'],
            callbacks=[early_stop_callback, lr_logger],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            gradient_clip_val=config.get('gradient_clip_val', 0.1),
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        # Evaluation
        test_data = TimeSeriesDataSet.from_dataset(training_data, test_processed, predict=True, stop_randomization=True)
        test_dataloader = test_data.to_dataloader(train=False, batch_size=config['batch_size'])

        predictions = model.predict(test_dataloader)

        # Convert predictions to class labels
        predicted_classes = predictions.argmax(dim=-1)

        # Calculate accuracy
        actuals = torch.cat([y for x, y in iter(test_dataloader)])
        accuracy = (predicted_classes.view(-1) == actuals.view(-1)).float().mean()

        logging.info(f'Test Accuracy: {accuracy:.4f}')

        save_and_display_results_classification(actuals.numpy(), predicted_classes.numpy(), subfolder, dataset='test')

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())

    logging.info("Main completed")


if __name__ == "__main__":
    config_path = '../../config/config.yaml'
    main(config_path)