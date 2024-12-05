import pandas as pd

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
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Drop missing values
    data.dropna(inplace=True)

    # Keep relevant columns
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    print("Data cleaning complete.")
    return data
