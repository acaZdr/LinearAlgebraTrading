import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def remove_outliers(y_true, y_pred, percentage=10):
    """Remove the top percentage of outliers based on residuals."""
    residuals = np.abs(y_true - y_pred)
    threshold = np.percentile(residuals, 100 - percentage)
    mask = residuals <= threshold
    return y_true[mask], y_pred[mask]

def plot_predictions(y_true, y_pred, title):
    """Plot the actual and predicted values."""
    plt.figure(figsize=(14, 7), dpi=300)
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

# Read the data
y_true = read_csv('test_actuals.csv').values.flatten()
y_pred = read_csv('test_predictions.csv').values.flatten()

# Remove outliers
y_true_filtered, y_pred_filtered = remove_outliers(y_true, y_pred)

# Plot the filtered data
plot_predictions(y_true_filtered, y_pred_filtered, 'Actual vs. Predicted Values')