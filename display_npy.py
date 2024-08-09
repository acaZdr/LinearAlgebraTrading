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


display_npy_file('results/test_actuals.npy')
display_npy_file('results/test_predictions.npy')