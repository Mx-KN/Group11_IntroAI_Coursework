from pathlib import Path

from linear_regression_v2 import process_and_train_binary_classification
from test import process_and_train_binary_classification

from random_forest import process_and_train_random_forest



def main():
    """
    Main workflow to run the daily linear regression classification approach.
    """
    # Define the file path for the dataset
    PROJECT_ROOT = Path(__file__).resolve().parent
    RAW_FILE = PROJECT_ROOT / "../data/raw/btc_usd.csv"

    print("Running daily linear regression classification...")
    try:
      
        process_and_train_random_forest(RAW_FILE)

        process_and_train_binary_classification(
        RAW_FILE,
        test_size=0.3,
        random_state=42,
        lookahead_days=30
    )

    except Exception as e:
        print(f"Error running daily linear regression: {e}")


if __name__ == "__main__":
    main()
