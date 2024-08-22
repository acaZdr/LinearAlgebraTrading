import logging
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from logic.models.abstract_model import set_up_folders, save_experiment_results
from logic.models.lstm_dyn_attention_classification import LSTMModel, evaluate_dollar_difference, preprocess_data, \
    CryptoDataset
from src.data_preprocessing.data_importer import import_data
from src.utils.config_loader import load_config
from src.utils.data_saving_and_displaying import save_and_display_results_classification

project_root, subfolder = set_up_folders()
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(device.type)



class EnsembleLSTMModel(nn.Module):
    def __init__(self, num_models, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0, use_attention=True):
        super(EnsembleLSTMModel, self).__init__()
        self.models = nn.ModuleList([
            LSTMModel(input_dim, hidden_dim, num_layers, num_classes, dropout, use_attention)
            for _ in range(num_models)
        ])

    def forward(self, x, volatility, volume):
        outputs = []
        attentions = []
        for model in self.models:
            output, attention = model(x, volatility, volume)
            outputs.append(output)
            attentions.append(attention)

        # Use voting instead of averaging
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        ensemble_attention = torch.stack(attentions, dim=0).mean(dim=0)

        return ensemble_output, ensemble_attention

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def train_ensemble_model(ensemble_model, train_loader, val_loader, criterion, optimizers, schedulers, num_epochs, patience=5):
    ensemble_model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loss = 0
        for X_batch, volatility_batch, volume_batch, y_batch in train_loader:
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(device), volume_batch.to(device), y_batch.to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            y_pred, _ = ensemble_model(X_batch, volatility_batch, volume_batch)

            assert not torch.isnan(y_pred).any(), "NaN values found in model output"
            loss = criterion(y_pred, y_batch)
            loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            train_loss += loss.item()

        val_loss = evaluate_ensemble_model(ensemble_model, val_loader, criterion)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        for scheduler in schedulers:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ensemble_model.save(os.path.join(subfolder, 'best_ensemble_lstm_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break

def evaluate_ensemble_model(ensemble_model, data_loader, criterion):
    ensemble_model.eval()
    total_loss = 0
    all_y_true = []
    all_y_pred = []

    with (torch.no_grad()):
        for X_batch, volatility_batch, volume_batch, y_batch in data_loader:
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(device), volume_batch.to(device), y_batch.to(device)
            y_pred = None #samo da ne bi bilo errora

            model = ensemble_model(X_batch, volatility_batch, volume_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())

    average_loss = total_loss / len(data_loader)

    return average_loss, all_y_true, all_y_pred

def evaluate_dollar_difference_ensemble(ensemble_model, data_loader, scaler_y, device):
    ensemble_model.eval()
    total_abs_error = 0
    count = 0

    with torch.no_grad():
        for X_batch, volatility_batch, volume_batch, y_batch in data_loader:
            X_batch, volatility_batch, volume_batch, y_batch = X_batch.to(device), volatility_batch.to(device), volume_batch.to(device), y_batch.to(device)
            y_pred, _ = ensemble_model(X_batch, volatility_batch, volume_batch)

            y_pred = y_pred[-len(y_batch):, :]

            y_pred_np = y_pred.cpu().numpy()
            y_batch_np = y_batch.cpu().numpy().reshape(-1, 1)

            y_pred_unscaled = scaler_y.inverse_transform(y_pred_np)
            y_batch_unscaled = scaler_y.inverse_transform(y_batch_np)

            total_abs_error += np.sum(np.abs(y_pred_unscaled - y_batch_unscaled))
            count += len(y_batch)

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

        # Define paths for datasets
        datasets = {
            'train': [os.path.join(project_root, 'data', path) for path in config['train_data']],
            'val': [os.path.join(project_root, 'data', path) for path in config['val_data']],
            'test': [os.path.join(project_root, 'data', path) for path in config['test_data']]
        }

        processed_data = {}
        data_loaders = {}
        scaler_X = None
        scaler_y = None
        scaler_volatility = None
        pca = None

        # Process each dataset (train, val, test)
        for dataset_name, data_path in datasets.items():
            logging.info(f"Processing {dataset_name} dataset from {data_path}")
            data = import_data(data_path, limit=config.get('data_limit', None))
            logging.info(f"Data imported for {dataset_name}, shape: {data.shape}")

            if dataset_name == 'train':
                processed_data[
                    dataset_name], preprocessed_volatility, preprocessed_volume, scaler_X, scaler_y, scaler_volatility, scaler_volume, pca = preprocess_data(
                    data, config, fit=False)
            else:
                processed_data[
                    dataset_name], preprocessed_volatility, preprocessed_volume, _, _, _, _, _ = preprocess_data(data,
                                                                                                                 config,
                                                                                                                 scaler_X,
                                                                                                                 scaler_y,
                                                                                                                 scaler_volatility,
                                                                                                                 scaler_volume,
                                                                                                                 pca,
                                                                                                                 fit=True)

            logging.info(f"Data preprocessed for {dataset_name}, shape: {processed_data[dataset_name].shape}")

            dataset = CryptoDataset(processed_data[dataset_name], preprocessed_volatility, preprocessed_volume,
                                    seq_length=config['seq_length'])
            data_loaders[dataset_name] = DataLoader(dataset, batch_size=config['batch_size'],
                                                    shuffle=(dataset_name == 'train'))
            logging.info(f"DataLoader created for {dataset_name}")
        input_dim = processed_data['train'].shape[1] - 1
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config.get('dropout', 0)
        use_attention = config['use_attention']
        num_classes = 3

        num_models = config.get('num_ensemble_models', 5)
        ensemble_model = EnsembleLSTMModel(num_models, input_dim, hidden_dim, num_layers, num_classes, dropout,
                                           use_attention)
        logging.info(f"Ensemble model initialized with {num_models} LSTM models")

        criterion = nn.CrossEntropyLoss()

        optimizers = [
            torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0))
            for model in ensemble_model.models]
        schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) for
                      optimizer in optimizers]

        logging.info(f"Loss function, optimizer, and scheduler initialized. Learning rate: {config['learning_rate']}")

        # Train the ensemble model
        logging.info("Starting ensemble model training")
        start_time = time.time()
        train_ensemble_model(ensemble_model, data_loaders['train'], data_loaders['val'], criterion, optimizers,
                             schedulers, config['num_epochs'])
        end_time = time.time()
        training_time = end_time - start_time
        avg_time_per_epoch = training_time / config['num_epochs']
        logging.info("Ensemble model training completed")

        # Evaluate the ensemble model on the test set
        logging.info("Starting ensemble model evaluation on test set")
        ensemble_model.load(os.path.join(subfolder, 'best_ensemble_lstm_model.pth'))
        test_loss, test_actuals, test_predictions = evaluate_ensemble_model(ensemble_model, data_loaders['test'], criterion)
        print(f'Ensemble Test Loss: {test_loss:.6f}')

        # Save and display results
        #logging.info("Saving and displaying ensemble results")
        #save_and_display_results(test_actuals, test_predictions, subfolder)
        average_dollar_difference = evaluate_dollar_difference(ensemble_model, data_loaders['test'], scaler_y, device)
        print(f'Ensemble Average Dollar Difference: ${average_dollar_difference:.2f}')


        save_experiment_results(
            training_time, avg_time_per_epoch, test_loss, average_dollar_difference,
            config.get('data_limit', 'N/A'), config.get('use_pca', False), csv_path
        )

        save_and_display_results_classification(test_actuals, test_predictions, subfolder)


    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc()

    logging.info("Main completed")


if __name__ == "__main__":
    config_path = '../../config/config.yaml'
    main(config_path)