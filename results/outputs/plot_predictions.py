import matplotlib.pyplot as plt
from pandas import read_csv


def plot_predictions(y_true, y_pred, title):
    """Plot the actual and predicted values."""
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

y_true = read_csv('test_actuals.csv')
y_pred = read_csv('test_predictions.csv')
plot_predictions(y_true, y_pred, 'Actual vs. Predicted Values')