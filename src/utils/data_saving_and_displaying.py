import os
import numpy as np
import pandas as pd


def display_npy_file(file_path):
    # Load the .npy file
    data = np.load(file_path)

    # Convert to a DataFrame for better readability
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)

    # Optionally, save to a CSV file for viewing in an editor
    csv_file_path = file_path.replace('.npy', '.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")
    # print mean and std
    print(f"Mean: {np.mean(data)}")
    print(f"Std: {np.std(data)}")


def save_and_display_results(test_actuals, test_predictions, subfolder):
    """Save the actual and predicted values, and convert them to CSV figures."""
    actuals_path = os.path.join(subfolder, 'test_actuals.npy')
    predictions_path = os.path.join(subfolder, 'test_predictions.npy')

    # Save as .npy figures
    np.save(actuals_path, np.array(test_actuals))
    np.save(predictions_path, np.array(test_predictions))

    # Convert .npy to .csv
    display_npy_file(actuals_path)
    display_npy_file(predictions_path)