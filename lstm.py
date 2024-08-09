import numpy as np
import torch
import torch.nn as nn
from ta import add_all_ta_features
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import yaml
import os
import logging
import traceback
from data_preprocessing import import_data
import torch.optim.lr_scheduler as lr_scheduler

A# Set up the results subfolder
subfolder = 'results'
if not os.path.exists(subfolder):
    os.makedirs(subfolder)


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

    # Custom dataset


class CryptoDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length, :-1],
                self.data[idx + self.seq_length - 1, -1])


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def choose_n_components(X_scaled, variance_threshold=0.95):
    pca = PCA().fit(X_scaled)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumulative_variance_ratio >= variance_threshold)[0][0] + 1

    # Optionally, plot the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.axvline(x=n_components, linestyle='--', color='r',
                label=f'n_components for {variance_threshold * 100}% variance')
    plt.legend()
    # plt.show()

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
    X = data.drop(columns=[target]).values
    y = data[target].values

    # Check for any remaining NaN values in features
    if np.isnan(X).any():
        raise ValueError("Input X contains NaN values after replacing infinities")

    # Standardize the features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Check if PCA components file exists
    pca_file = os.path.join(subfolder, 'pca_components.npy')
    pca_mean_file = os.path.join(subfolder, 'pca_mean.npy')
    if os.path.exists(pca_file) and os.path.exists(pca_mean_file):
        pca_components = np.load(pca_file)
        pca_mean = np.load(pca_mean_file)
        n_components = pca_components.shape[0]
        pca = PCA(n_components=n_components)
        pca.components_ = pca_components
        pca.mean_ = pca_mean
        X_pca = pca.transform(X_scaled)
    else:
        # Perform PCA on all features
        n_components = choose_n_components(X_scaled, variance_threshold=0.95)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        np.save(pca_file, pca.components_)
        np.save(pca_mean_file, pca.mean_)
        logging.info(f"PCA components and mean saved as '{pca_file}' and '{pca_mean_file}'")

    final_features = [f"Component_{i + 1}" for i in range(n_components)]
    logging.info(f"Final features used by the NN: {final_features}")

    # Scale the target
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Ensure y_scaled is a 1D array

    # Ensure the feature and target arrays have the same number of rows
    if X_pca.shape[0] != y_scaled.shape[0]:
        raise ValueError(f"Feature and target arrays have mismatched sizes: {X_pca.shape[0]} vs {y_scaled.shape[0]}")

    return np.hstack((X_pca, y_scaled.reshape(-1, 1))), pca


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
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

        if patience_counter >= patience:
            print("Early stopping triggered")
            break


def evaluate_dollar_difference(model, data_loader, scaler_y, device):
    model.eval()
    total_abs_error = 0
    count = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, _ = model(X_batch)

            # Convert predictions and targets back to the original scale
            y_pred_unscaled = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            y_batch_unscaled = scaler_y.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1))

            # Calculate the absolute error
            total_abs_error += np.sum(np.abs(y_pred_unscaled - y_batch_unscaled))
            count += len(y_batch)

    average_dollar_diff = total_abs_error / count
    return average_dollar_diff


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Process training, validation, and test data
        datasets = {'train': config['train_data_path'],
                    'val': config['val_data_path'],
                    'test': config['test_data_path']}

        processed_data = {}
        data_loaders = {}
        scaler_y = None

        for dataset_name, data_path in datasets.items():
            # Load data
            data = import_data(data_path, limit=config.get('data_limit'))
            logging.info(f"{dataset_name.capitalize()} data loaded successfully. Shape: {data.shape}")

            # Log data info
            logging.info(f"Columns in the {dataset_name} dataset: {data.columns.tolist()}")
            logging.info(f"Data types: \n{data.dtypes}")

            # Check for non-numeric columns
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_columns) > 0:
                data = data.drop(columns=non_numeric_columns)

            logging.info(f"First few rows of the {dataset_name} data: \n{data.head()}")
            logging.info(f"{dataset_name.capitalize()} data description: \n{data.describe()}")

            # Check for any initial NaN or infinite values
            if data.isna().any().any():
                logging.warning(f"Initial {dataset_name} data contains NaN values")

            # Convert data to float for isinf check
            numeric_data = data.astype(np.float64)
            if np.isinf(numeric_data.values).any():
                logging.warning(f"Initial {dataset_name} data contains infinite values")

            # Preprocess data
            processed_data[dataset_name], pca = preprocess_data(data, config)
            logging.info(
                f"{dataset_name.capitalize()} data preprocessed successfully. Shape: {processed_data[dataset_name].shape}")

            # Create datasets and dataloaders
            dataset = CryptoDataset(processed_data[dataset_name], seq_length=config['seq_length'])
            data_loaders[dataset_name] = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

            if dataset_name == 'train':
                # Extract the target scaler from the training data
                _, scaler_y = preprocess_data(data, config)

        # Initialize the model
        input_dim = processed_data['train'].shape[1] - 1
        model = LSTMModel(input_dim=input_dim, hidden_dim=config['hidden_dim'],
                          num_layers=config['num_layers'], output_dim=1, dropout=config['dropout'])

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Train the model
        train_model(model, data_loaders['train'], data_loaders['val'], criterion, optimizer, scheduler,
                    config['num_epochs'], torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Evaluate the model on the test set
        model.load_state_dict(torch.load(os.path.join(subfolder, 'best_lstm_model.pth')))
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in data_loaders['test']:
                X_batch, y_batch = X_batch.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), y_batch.to(
                    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                y_pred, _ = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                test_loss += loss.item()
        test_loss /= len(data_loaders['test'])
        logging.info(f'Test Loss: {test_loss:.4f}')
        print(f"y_pred shape: {y_pred.shape}")
        print(f"scaler_y mean shape: {scaler_y.mean_.shape}")  # This assumes `scaler_y` is a StandardScaler

        average_dollar_difference = evaluate_dollar_difference(model, data_loaders['test'], scaler_y,
                                                               torch.device(
                                                                   'cuda' if torch.cuda.is_available() else 'cpu'))
        logging.info(f'Average Dollar Difference: ${average_dollar_difference:.2f}')

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    config_path = 'config.yaml'
    main(config_path)
