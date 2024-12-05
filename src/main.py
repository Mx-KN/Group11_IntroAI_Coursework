from pathlib import Path
from data_preprocessing import load_data, clean_data
from feature_engineering import add_technical_indicators

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FILE = PROJECT_ROOT / "../data/raw/btc_usd.csv"
PROCESSED_DATA_FILE = PROJECT_ROOT / "../data/processed/final_btc_with_indicators.csv"

def main():
    """
    Main workflow to load, clean, and add technical indicators to the data.
    """
    # Step 1: Load raw data
    print(f"Loading raw data from {RAW_FILE}...")
    raw_data = load_data(RAW_FILE)
    if raw_data is None:
        print("Failed to load raw data. Exiting...")
        return

    # Step 2: Clean data
    print("\nCleaning data...")
    cleaned_data = clean_data(raw_data)

    # Step 3: Add technical indicators
    print("\nAdding technical indicators...")
    final_data = add_technical_indicators(cleaned_data)


    # drop rows where there isn't enough histroic data for TA features
    final_data = final_data.dropna()

    # Step 4: Display and optionally save the final DataFrame
    print("\nFinal DataFrame (First 5 Rows):")
    print(final_data.head())
    print(final_data.describe())



if __name__ == "__main__":
    main()
