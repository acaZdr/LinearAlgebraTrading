import pandas as pd


def combine_csv_files(file_paths, limit=None) -> pd.DataFrame:
    data_frames = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if limit:
            df = df.head(limit)
        data_frames.append(df)

    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


def import_data(file_paths, limit=None):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    combined_data = combine_csv_files(file_paths, limit)

    df = combined_data.iloc[:, :6]  # Select only the first 6 columns
    df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']

    df['date'] = pd.to_datetime(df['date'], unit='ms')

    return df

