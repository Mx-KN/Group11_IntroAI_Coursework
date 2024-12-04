import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_data(file_path):
    """
    Load the Bitcoin dataset into a Pandas DataFrame.
    """
    print("Loading Bitcoin data...")
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(data):
    """
    Clean the Bitcoin dataset:
    - Convert Date to datetime
    - Handle missing values
    - Keep relevant columns
    """
    print("Cleaning Bitcoin data...")
    # convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # drop missing values
    data.dropna(inplace=True)

    # drop irrelevant columns
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    print("Data cleaning complete.")
    return data

if __name__ == "__main__":
    raw_file_path = os.path.join(PROJECT_ROOT, "../data/raw/btc_usd.csv")

    btc_data = load_data(raw_file_path)

    if btc_data is not None:
        print("\nRaw DataFrame (First 5 Rows):")
        print(btc_data.head())

        cleaned_data = clean_data(btc_data)

        print("\nCleaned DataFrame (First 5 Rows):")
        print(cleaned_data.head())
