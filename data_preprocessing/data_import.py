import pandas as pd


def import_data(csv_path: str, limit=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.iloc[:, :6]  # Select only the first 6 columns
    df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']

    df['date'] = pd.to_datetime(df['date'], unit='ms')

    if limit:
        df = df.head(limit)

    return df
