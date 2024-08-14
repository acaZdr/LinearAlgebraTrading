import numpy as np
import torch
import torch.nn as nn
from ta import add_all_ta_features
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import os
import logging
import traceback
import torch.optim.lr_scheduler as lr_scheduler

from logic.models.abstract_model import choose_n_components, evaluate_dollar_difference
from src.data_preprocessing.data_importer import import_data
from src.utils.config_loader import load_config
from src.utils.data_saving_and_displaying import save_and_display_results
from torch.utils.data import Dataset

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
subfolder = os.path.join(project_root, 'results', 'outputs')
if not os.path.exists(subfolder):
    os.makedirs(subfolder)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class CryptoDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length, :-1],
                self.data[idx + self.seq_length - 1, -1])


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)
        return context_vector, attention_weights


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        context_vector, attention_weights = self.attention(lstm_out)
        out = self.fc(context_vector)
        return out.view(-1, 1), attention_weights


def preprocess_data(data, config, scaler_X=None, scaler_y=None, pca=None, fit=False):
    # Shift the target variable
    target = config['target']
    data['target'] = data[target].shift(-1)  # Shift the target to predict next minute's close

    # Calculate lookback technical indicators
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

    logging.info(f"Number of features before PCA: {len(feature_columns)}")

    X = data[feature_columns].values
    y = data['target'].values

    if not fit:
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        if config.get('use_pca', False):
            logging.info("PCA is enabled. Applying PCA...")
            pca = PCA(n_components=choose_n_components(X_scaled,
                                                       variance_threshold=config.get('variance_threshold', 0.95)))
            X_scaled = pca.fit_transform(X_scaled)
            logging.info(f"PCA applied. Number of components: {pca.n_components_}")
            logging.info(f"Variance explained by PCA: {sum(pca.explained_variance_ratio_):.4f}")
        else:
            logging.info("PCA is not enabled.")
    else:
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()

        if pca is not None:
            X_scaled = pca.transform(X_scaled)
            logging.info(f"PCA transform applied. Number of components: {pca.n_components_}")

    logging.info(f"Number of features after preprocessing: {X_scaled.shape[1]}")

    # Ensure no NaN values
    assert not np.isnan(X_scaled).any(), "NaN values found in features"
    assert not np.isnan(y_scaled).any(), "NaN values found in target"

    return np.hstack((X_scaled, y_scaled.reshape(-1, 1))), scaler_X, scaler_y, pca


def train_model(model: nn.Module, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred, _ = model(X_batch)  # Extract the predictions
            loss = criterion(y_pred.squeeze(), y_batch)  # Apply squeeze only to y_pred
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch)  # Extract the predictions
                loss = criterion(y_pred.squeeze(), y_batch)  # Apply squeeze only to y_pred
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Step the scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(subfolder, 'best_lstm_model.pth'))
        else:
            patience_counter += 1

        # if patience_counter >= patience:
        #     print("Early stopping triggered")
        #     break


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        logging.info("Starting main function")
        logging.info(f"Configuration loaded from: {config_path}")

        # Process training, validation, and test data
        datasets = {
            'train': [os.path.join(project_root, 'data', path) for path in config['train_data']],
            'val': [os.path.join(project_root, 'data', path) for path in config['val_data']],
            'test': [os.path.join(project_root, 'data', path) for path in config['test_data']]
        }

        processed_data = {}
        data_loaders = {}
        scaler_X = None
        scaler_y = None
        pca = None
        fit = False

        for dataset_name, data_path in datasets.items():
            logging.info(f"Processing {dataset_name} dataset from {data_path}")
            # Load and preprocess data
            data = import_data(data_path, limit=config['data_limit'])
            logging.info(f"Data imported for {dataset_name}, shape: {data.shape}")

            processed_data[dataset_name], scaler_X, scaler_y, pca = preprocess_data(data, config, scaler_X, scaler_y,
                                                                                    pca, fit)
            logging.info(f"Data preprocessed for {dataset_name}, shape: {processed_data[dataset_name].shape}")

            fit = True
            dataset = CryptoDataset(processed_data[dataset_name], seq_length=config['seq_length'])
            data_loaders[dataset_name] = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
            logging.info(f"DataLoader created for {dataset_name}")

        # Initialize the model
        input_dim = processed_data['train'].shape[1] - 1
        logging.info(f"Input dimension: {input_dim}")
        model = LSTMModel(input_dim=input_dim, hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                          output_dim=1, dropout=config['dropout'])
        logging.info(f"Model initialized with hidden_dim: {config['hidden_dim']}, num_layers: {config['num_layers']}")

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        logging.info(f"Loss function, optimizer, and scheduler initialized. Learning rate: {config['learning_rate']}")

        # Train the model
        logging.info("Starting model training")
        train_model(model, data_loaders['train'], data_loaders['val'], criterion, optimizer, scheduler,
                    config['num_epochs'])
        logging.info("Model training completed")

        # Evaluate the model on the test set
        logging.info("Starting model evaluation on test set")
        model.load_state_dict(torch.load(os.path.join(subfolder, 'best_lstm_model.pth')))
        model.eval()
        test_loss = 0
        test_actuals = []
        test_predictions = []
        with torch.no_grad():
            for X_batch, y_batch in data_loaders['test']:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                test_loss += loss.item()
                test_actuals.extend(y_batch.cpu().numpy())
                test_predictions.extend(y_pred.squeeze().cpu().numpy())
        test_loss /= len(data_loaders['test'])
        logging.info(f'Test Loss: {test_loss:.4f}')

        # Save and display results
        logging.info("Saving and displaying results")
        save_and_display_results(test_actuals, test_predictions, subfolder)
        logging.info(f"Results saved in {subfolder}")

        # Calculate and log average dollar difference
        logging.info("Calculating average dollar difference")
        average_dollar_difference = evaluate_dollar_difference(model, data_loaders['test'], scaler_y, device)
        logging.info(f'Average Dollar Difference: ${average_dollar_difference:.2f}')

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc()

    logging.info("Main completed")


if __name__ == "__main__":
    config_path = '../../config/config.yaml'
    main(config_path)
