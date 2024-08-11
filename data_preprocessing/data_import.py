import pandas as pd
import csv


def combine_csv_files(file_paths, limit=None) -> pd.DataFrame:
    data_frames = []

    for file_path in file_paths:
        try:
            # Read CSV without headers and assign numeric column names
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                first_row = next(reader)
                num_columns = len(first_row)

            column_names = [f'col_{i}' for i in range(num_columns)]

            df = pd.read_csv(file_path, header=None, names=column_names)

            if limit is not None:
                df = df.head(limit)
            data_frames.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not data_frames:
        raise ValueError("No files were successfully loaded.")

    # Check that all dataframes have the same number of columns
    num_columns = data_frames[0].shape[1]
    for df in data_frames[1:]:
        if df.shape[1] != num_columns:
            raise ValueError("Not all CSV files have the same number of columns.")

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

