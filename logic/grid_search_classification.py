import yaml
import itertools
import os
from logic.models.lstm_dyn_attention_classification import main  # Assuming your main training function is in main.py
from src.utils.config_loader import load_config


def grid_search(base_config_path, output_dir):
    # Load the base configuration
    base_config = load_config(base_config_path)

    # Define the hyperparameter grid
    param_grid = {
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3, 4],
        'learning_rate': [0.0001, 0.00005, 0.00001],
        'dropout': [0.2, 0.3, 0.4],
        'weight_decay': [0.0005, 0.001, 0.002],
        'batch_size': [32, 64, 128],
        'seq_length': [30, 60, 90]
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
        config_filename = f"config_{i + 1}.yaml"
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