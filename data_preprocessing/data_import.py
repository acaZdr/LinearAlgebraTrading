import pandas as pd


def import_data(csv_path: str, limit=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[:limit, :6]
    else:
        df = df.iloc[:, :6]
    df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']

    df['date'] = pd.to_datetime(df['date'], unit='ms')
    return df
