# LinearAlgebraTrading

## Overview

LinearAlgebraTrading is a cryptocurrency trading project that applies linear algebra techniques, specifically Principal Component Analysis (PCA), to preprocess crypto price data. The processed data is then used to train models, including an LSTM with an attention mechanism and a linear regression model, to predict future prices.

## Project Structure

- **config/**: Contains configuration files for the project.
  - `config.yaml`: Main configuration file that needs to be edited to match the CSV datasets.
  
- **logic/**: Contains the implementation of the models.
  - `lstm.py`: LSTM model with an attention mechanism for price prediction.
  - `lin_regression.py`: Linear regression model for price prediction.
  
- **results/**: Stores results, logs, and other output files generated by the models.
  
- **scripts/**: Contains utility scripts.
  - `binance_scraper.py`: Script to download cryptocurrency data from Binance.
  
- **src/**: Includes data preprocessing and utility functions.
  - `data_preprocessing/`: Functions for data preprocessing and transformation.
  - `utils/`: Miscellaneous utility functions.

## Setup and Instructions

### 1. Download the Data

First, download the required cryptocurrency data using the Binance scraper:

```bash
python scripts/binance_scraper.py
```

This will download the data and save it as CSV files in a specified directory.

### 2. Configure the Project

Edit the `config/config.yaml` file to specify the paths to your CSV datasets. The current configuration is set to:

- `BTCUSDT-1s-2023-11.csv`
- `BTCUSDT-1s-2023-12.csv`
- `BTCUSDT-1s-2024-01.csv`

Update these paths if necessary.

### 3. Run the Models

To run the models, choose one of the following commands:

- For the LSTM model with attention:

```bash
python logic/models/lstm.py
```

- For the linear regression model:

```bash
python logic/models/lin_regression.py
```

The results will be saved in the `results/` directory.

## Additional Information

This project uses PCA to reduce the dimensionality of cryptocurrency price data before applying models. The LSTM model incorporates an attention mechanism to improve the prediction accuracy by focusing on relevant features.
