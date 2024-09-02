import yaml
import itertools
import os
from logic.models.lstm_dyn_attention_classification import main
from src.utils.config_loader import load_config


def grid_search(base_config_path, output_dir):
    # Load the base configuration
    base_config = load_config(base_config_path)

    # Define the hyperparameter grid
    og_param_grid = {
        'hidden_dim': [64, 128, 256],#0,96,192
        'num_layers': [2, 3],#0,48
        'learning_rate': [0.0001, 0.00001],#0,24
        'dropout': [0.2,0.4],#0,12
        'weight_decay': [0.0005, 0.001],#0,6
        'batch_size': [32, 64, 128], #0,3,6
        'seq_length': [30, 60, 90] #0,1,2
    }

    second_param_grid = {
        'hidden_dim': [64,80,96],
        'num_layers': [2,3],
        'learning_rate': [0.0001],
        'dropout': [0.2],
        'weight_decay': [0.0005],
        'batch_size': [128],
        'seq_length': [30],
        'num_epochs': [10]
    }

    third_param_grid = {
        'hidden_dim': [32,64,128],
        'num_layers': [2],
        'learning_rate': [0.0001],
        'dropout': [0.25],
        'weight_decay': [0.0005],
        'batch_size': [128],
        'seq_length': [15,30,45],
        'num_epochs': [30]
    }

    final_param_grid = {
        'hidden_dim': [128],
        'num_layers': [2],
        'learning_rate': [0.0001],
        'dropout': [0.25],
        'weight_decay': [0.0005],
        'batch_size': [128],
        'seq_length': [45],
        'num_epochs': [200],
        'use_pca': [True, False]
    }

    param_grid = {
        'hidden_dim': [128],
        'num_layers': [2],
        'learning_rate': [0.0001],
        'dropout': [0.2],
        'weight_decay': [0.0005],
        'batch_size': [256],
        'seq_length': [5,20,40],
        'volatility_window_size': [5,10,20,50,100],
        'num_epochs': [50],
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all hyperparameter combinations
    for i, params in enumerate(hyperparameter_combinations):
        print(f"Running combination {i + 1}/{len(hyperparameter_combinations)}")

        # Update the base configuration with the current hyperparameters
        current_config = base_config.copy()
        current_config.update(params)

        # Create a unique config file for this run
        config_filename = f"config_v2_{i + 1}.yaml"
        config_path = os.path.join(output_dir, config_filename)

        # Save the current configuration
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f)

        # Run the main training function with the current configuration
        main(config_path, grid_search_run=i)


if __name__ == "__main__":
    base_config_path = '../config/config.yaml'
    output_dir = '../results/outputs/grid_search_results'
    grid_search(base_config_path, output_dir)