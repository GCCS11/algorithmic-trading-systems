import pandas as pd


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    for df in [train, test]:
        df['Datetime'] = pd.to_datetime(df['Datetime'])

    train = train.sort_values('Datetime').reset_index(drop=True)
    test  = test.sort_values('Datetime').reset_index(drop=True)

    # Drop rows with missing OHLC
    train = train.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)
    test  = test.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)

    return train, test