import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ta import add_all_ta_features
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import logging
import traceback
import torch.optim.lr_scheduler as lr_scheduler

from logic.models.abstract_model import set_up_folders, save_experiment_results
from src.data_preprocessing.data_importer import import_data
from src.utils.config_loader import load_config
from src.utils.data_saving_and_displaying import save_and_display_results, save_and_display_results_classification
from src.data_preprocessing.data_preprocessor import DataPreprocessor

from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

import time

project_root, subfolder = set_up_folders()
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class DynamicAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(DynamicAttention, self).__init__()
        self.feature_layer = nn.Linear(2, hidden_dim, bias=False)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, volatility, volume):
        # Combine volatility and volume
        features = torch.cat((volatility.unsqueeze(-1), volume.unsqueeze(-1)), dim=-1)
        dynamic_weights = torch.tanh(self.feature_layer(features))
        attention_weights = torch.softmax(self.attention(lstm_out * dynamic_weights).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        return context_vector, attention_weights


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0, use_attention=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Time-distributed LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)


        self.attention = DynamicAttention(hidden_dim)

        # Deeper fully connected layers with Leaky ReLU, Dropout, and BatchNorm
        self.fc_layers = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 4),

            nn.Linear(hidden_dim // 4, num_classes)
        )
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x, volatility, volume):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm1(x, (h0, c0))


        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_out, volatility, volume)
        else:
            # Uniform attention weights
            seq_len = lstm_out.size(1)
            uniform_attention_weights = torch.ones(lstm_out.size(0), seq_len, device=lstm_out.device) / seq_len

            # Compute context vector using uniform weights
            context_vector = torch.sum(uniform_attention_weights.unsqueeze(-1) * lstm_out, dim=1)
            attention_weights = uniform_attention_weights

        out = self.fc_layers(context_vector)
        #out = self.softmax(out)
        return out, attention_weights


class CryptoDataset(Dataset):
    def __init__(self, data, volatility, volume, seq_length):
        self.data = torch.FloatTensor(data[:, :-1])  # Exclude the last column (target)
        self.volatility = torch.FloatTensor(volatility)

        self.volume = torch.FloatTensor(volume)
        self.seq_length = seq_length
        self.labels = torch.LongTensor(self._create_labels(data[:, -1]))

        expected_length = len(self)
        if len(self.labels) > expected_length:
            self.labels = self.labels[:expected_length]

    def _create_labels(self, target):
        # Calculate the Simple Moving Average (SMA) over the last 'seq_length' time points
        sma = np.convolve(target, np.ones(self.seq_length) / self.seq_length, mode='valid')

        # Calculate the difference between the current target value and the SMA
        diff_to_sma = target[-len(sma):] - sma  # Adjust to match the size of the SMA array

        # Calculate the 33rd and 67th percentiles
        lower_bound = np.percentile(diff_to_sma, 33)
        upper_bound = np.percentile(diff_to_sma, 67)

        # Initialize labels array with zeros
        labels = np.zeros_like(diff_to_sma, dtype=int)

        # Label assignment based on percentile ranges
        labels[diff_to_sma > upper_bound] = 2  # Top 33% (Bullish)
        labels[(diff_to_sma >= lower_bound) & (diff_to_sma <= upper_bound)] = 1  # Middle 33% (Neutral)
        labels[diff_to_sma < lower_bound] = 0  # Bottom 33% (Bearish)

        # print count of each label [0,1,2]
        logging.info(f"Label counts: {np.bincount(labels)}")

        return labels

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length],
                self.volatility[idx:idx + self.seq_length],
                self.volume[idx:idx + self.seq_length],
                self.labels[idx])


def add_custom_ta_features(data):
    # MACD
    macd = MACD(close=data['Close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()

    # EMA
    data['ema_9'] = EMAIndicator(close=data['Close'], window=9).ema_indicator()
    data['ema_21'] = EMAIndicator(close=data['Close'], window=21).ema_indicator()
    data['ema_50'] = EMAIndicator(close=data['Close'], window=50).ema_indicator()
    data['ema_200'] = EMAIndicator(close=data['Close'], window=200).ema_indicator()

    # RSI
    data['rsi_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
    data['rsi_21'] = RSIIndicator(close=data['Close'], window=21).rsi()

    # Bollinger Bands
    bb = BollingerBands(close=data['Close'])
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()
    data['bb_mid'] = bb.bollinger_mavg()
    data['bb_width'] = (data['bb_high'] - data['bb_low']) / data['bb_mid']

    # On-Balance Volume
    data['obv'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()

    # Price rate of change
    data['price_roc'] = data['Close'].pct_change(periods=12)

    return data


def calculate_volatility(data, window_size=20):
    data['log_return'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
    data['volatility'] = data['log_return'].rolling(window=window_size).std()
    return data['volatility'].dropna()

def compute_grad_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def time_series_cross_validation(data, n_splits=10, train_size=6, val_size=2, test_size=2):
    total_size = len(data)
    segment_size = total_size // n_splits

    for i in range(n_splits - (train_size + val_size + test_size) + 1):
        train_start = i * segment_size
        train_end = (i + train_size) * segment_size
        val_end = (i + train_size + val_size) * segment_size
        test_end = (i + train_size + val_size + test_size) * segment_size

        train_data = data[train_start:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:test_end]

        yield train_data, val_data, test_data


def aggregate_and_save_cv_results(cv_results, subfolder):
    all_test_actuals = []
    all_test_predictions = []
    avg_test_loss = 0

    for result in cv_results:
        all_test_actuals.extend(result['test_actuals'])
        all_test_predictions.extend(result['test_predictions'])
        avg_test_loss += result['test_loss']

    avg_test_loss /= len(cv_results)

    # Save and display aggregated results
    save_and_display_results_classification(all_test_actuals, all_test_predictions, subfolder,
                                            dataset='test_aggregated')

    # Save average test loss
    with open(os.path.join(subfolder, 'cv_results.txt'), 'w') as f:
        f.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    logging.info(f"Aggregated cross-validation results saved. Average Test Loss: {avg_test_loss:.6f}")


def preprocess_data(data: pd.DataFrame, config, data_preprocessor: DataPreprocessor):
    target = config['target']

    # # Calculate the difference in closing price
    data['Close_diff'] = data['Close'].diff()

    data['target'] = data[target].shift(-1)

    #
    # data['Average_Close_diff'] = data['Close_diff'].abs().rolling(window=26).mean()
    # # Shift the target to predict the next period's price change
    # data['target'] = (
    #         (data['Close_diff'].shift(-1) > 0.05 * data['Average_Close_diff']).astype(int) +
    #         (data['Close_diff'].shift(-1) < -0.05 * data['Average_Close_diff']).astype(int) * (-1) + 1
    # ).astype(int)

    # Remove the first row which will have NaN for Close_diff
    data = data.dropna().reset_index(drop=True)

    # data = add_custom_ta_features(data)
    # data = data.dropna().reset_index(drop=True)
    #
    # feature_columns = ['macd', 'macd_signal', 'macd_diff',
    #                    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    #                    'rsi_14', 'rsi_21',
    #                    'bb_high', 'bb_low', 'bb_mid', 'bb_width',
    #                    'obv', 'price_roc']

    data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
    data = data.dropna().reset_index(drop=True)

    look_ahead_indicators = ['trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a',
                             'trend_visual_ichimoku_b', 'trend_stc', 'trend_psar_up', 'trend_psar_down']

    feature_columns = [col for col in data.columns if col not in (
                ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target',
                 'Average_Close_diff'] + look_ahead_indicators)]

    logging.info(f"Number of features before PCA: {len(feature_columns)}")

    # Calculate volatility using the new method
    data['volatility'] = calculate_volatility(data, window_size=config.get('volatility_window_size', 20))

    # Drop the close column
    data = data.drop(columns=['Close'])

    # Drop rows with NaN values in volatility
    data = data.dropna().reset_index(drop=True)

    volatility = data['volatility']
    volume = data['Volume']

    X = data[feature_columns].values
    y = data['target'].values

    X_scaled, y_scaled, volatility_scaled, volume_scaled = data_preprocessor.fit_transform_data(X, y, volatility, volume,
                                                                                                subfolder)

    logging.info(f"Number of features after preprocessing: {X_scaled.shape[1]}")

    # Ensure no NaN values
    assert not np.isnan(X_scaled).any(), "NaN values found in features"
    assert not np.isnan(y_scaled).any(), "NaN values found in target"
    assert not np.isnan(volatility_scaled).any(), "NaN values found in volatility"

    return np.hstack((X_scaled, y_scaled.reshape(-1,
                                                 1))), volatility_scaled, volume_scaled



def train_model(model: nn.Module, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, volatility_batch, volume_batch, y_batch in train_loader:
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                device), volume_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred, _ = model(X_batch, volatility_batch, volume_batch)

            # Ensure no NaN values in model output
            assert not torch.isnan(y_pred).any(), "NaN values found in model output"
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        grad_norm = compute_grad_norms(model)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, volatility_batch, volume_batch, y_batch in val_loader:
                X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                    device), volume_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch, volatility_batch, volume_batch)

                # Ensure no NaN values in model output
                assert not torch.isnan(y_pred).any(), "NaN values found in model output"
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Grad Norm: {grad_norm:.6f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(subfolder, 'best_lstm_model.pth'))
        else:
            patience_counter += 1
        #
        # if patience_counter >= patience:
        #     logging.info("Early stopping triggered")
        #     break

        # completed_epochs += 1

    # end_time = time.time()
    # duration = end_time - start_time
    # average_time_per_epoch = duration / completed_epochs if completed_epochs > 0 else 0
    # print(f"Training completed in {duration:.2f} seconds")
    # print(f"Average time per epoch: {average_time_per_epoch:.2f} seconds")


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    actuals = []
    predictions = []
    with torch.no_grad():
        for inputs, volatility, volume, targets in data_loader:
            inputs, volatility, volume, targets = inputs.to(device), volatility.to(device), volume.to(device), targets.to(device)
            outputs, _ = model(inputs, volatility, volume)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            actuals.extend(targets.cpu().numpy())
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    return total_loss / len(data_loader), actuals, predictions


def evaluate_dollar_difference(model, data_loader, scaler_y, device):
    model.eval()
    total_abs_error = 0
    count = 0

    if not isinstance(scaler_y, StandardScaler):
        raise TypeError(f"Expected StandardScaler, but got {type(scaler_y)}")

    with torch.no_grad():
        for X_batch, volatility_batch, volume_batch, y_batch in data_loader:
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(
                device), volume_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch, volatility_batch, volume_batch)

            y_pred = y_pred[-len(y_batch):, :]

            y_pred_np = y_pred.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy().reshape(-1, 1)

            try:
                y_pred_unscaled = scaler_y.inverse_transform(y_pred_np)
                y_batch_unscaled = scaler_y.inverse_transform(y_batch_np)

                total_abs_error += np.sum(np.abs(y_pred_unscaled - y_batch_unscaled))
                count += len(y_batch)
            except ValueError as e:
                logging.error(f"Error in inverse transform: {str(e)}")
                logging.error(f"y_pred_np shape: {y_pred_np.shape}, y_batch_np shape: {y_batch_np.shape}")
                raise

    if count == 0:
        raise ValueError("No samples were processed")

    average_dollar_diff = total_abs_error / count
    return average_dollar_diff


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    csv_path = os.path.join(subfolder, 'times.csv')

    try:
        logging.info("Starting main function")
        logging.info(f"Configuration loaded from: {config_path}")

        # Load all data paths (train, val, test)
        all_data_paths = config['train_data'] + config['val_data'] + config['test_data']
        all_data = pd.concat([import_data(os.path.join(project_root, 'data', path)) for path in all_data_paths])
        all_data = all_data.sort_values('date').reset_index(drop=True)

        # Initialize data preprocessing
        scaler_type = config.get('scaler', 'standard')
        use_pca = config.get('use_pca', False)
        data_preprocessor = DataPreprocessor(scaler_type=scaler_type, use_pca=use_pca)

        # Preprocess all data
        all_processed_data, all_volatility, all_volume = preprocess_data(all_data, config, data_preprocessor)

        # Time series cross-validation
        cv_results = []
        for fold, (train_data, val_data, test_data) in enumerate(time_series_cross_validation(all_processed_data)):
            logging.info(f"Processing fold {fold + 1}")

            # Create datasets and data loaders for each fold
            train_dataset = CryptoDataset(train_data, all_volatility[:len(train_data)], all_volume[:len(train_data)], config['seq_length'])
            val_dataset = CryptoDataset(val_data, all_volatility[len(train_data):len(train_data)+len(val_data)],
                                        all_volume[len(train_data):len(train_data)+len(val_data)], config['seq_length'])
            test_dataset = CryptoDataset(test_data, all_volatility[len(train_data)+len(val_data):],
                                         all_volume[len(train_data)+len(val_data):], config['seq_length'])

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

            # Initialize model, criterion, optimizer, and scheduler
            input_dim = train_data.shape[1] - 1  # Exclude target column
            hidden_dim = config['hidden_dim']
            num_layers = config['num_layers']
            dropout = config.get('dropout', 0)
            use_attention = config['use_attention']
            model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=3,
                              dropout=dropout, use_attention=use_attention)
            logging.info(f"Model initialized with hidden_dim: {hidden_dim}, num_layers: {num_layers}, dropout: {dropout}")

            criterion = CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0))
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            logging.info(f"Loss function, optimizer, and scheduler initialized. Learning rate: {config['learning_rate']}")

            # Train the model for the current fold
            logging.info(f"Starting model training for fold {fold + 1}")
            start_time = time.time()
            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config['num_epochs'])
            end_time = time.time()
            training_time = end_time - start_time
            avg_time_per_epoch = training_time / config['num_epochs']
            logging.info(f"Model training completed for fold {fold + 1}")

            # Evaluate the model on the test set for the current fold
            logging.info(f"Starting model evaluation on test set for fold {fold + 1}")
            test_loss, test_actuals, test_predictions = evaluate_model(model, test_loader, criterion, device)

            # Save results for the current fold
            cv_results.append({
                'fold': fold + 1,
                'test_loss': test_loss,
                'test_actuals': test_actuals,
                'test_predictions': test_predictions
            })

            # Save and display results for this fold
            save_and_display_results_classification(test_actuals, test_predictions, subfolder, dataset=f'test_fold_{fold + 1}')

        # Aggregate and save results across all folds
        aggregate_and_save_cv_results(cv_results, subfolder)
        logging.info("Cross-validation completed")

        # Additional evaluations on the final test set (if required)
        model.load_state_dict(torch.load(os.path.join(subfolder, 'best_lstm_model.pth')))
        model.eval()

        test_loss = 0.0
        test_actuals = []
        test_predictions = []

        with torch.no_grad():
            for X_batch, volatility_batch, volume_batch, y_batch in test_loader:
                X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(device), volume_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch, volatility_batch, volume_batch)
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
                test_actuals.extend(y_batch.cpu().numpy())
                test_predictions.extend(torch.argmax(y_pred, dim=1).cpu().numpy())

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.6f}')

        save_and_display_results(test_actuals, test_predictions, subfolder)
        save_experiment_results(training_time, avg_time_per_epoch, test_loss, 0.0, config.get('data_limit', 'N/A'), config.get('use_pca', False), csv_path)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc()

    logging.info("Main completed")


if __name__ == "__main__":
    config_path = '../../config/config.yaml'
    main(config_path)
