import linear_regression_daily
from pathlib import Path
from linear_regression_daily import run_daily_linear_regression
from linear_regression_weekly_monthly import run_linear_regression_classification

def main():
    """
    Main workflow to run the daily linear regression classification approach.
    """
    # Define the file path for the dataset
    PROJECT_ROOT = Path(__file__).resolve().parent
    RAW_FILE = PROJECT_ROOT / "../data/raw/btc_usd.csv"

    print("Running daily linear regression classification...")
    try:
        # run_daily_linear_regression(RAW_FILE, threshold=0.6)

        # # Call the daily linear regression pipeline
        run_linear_regression_classification(RAW_FILE, threshold=0.4, sample_size=100)



    except Exception as e:
        print(f"Error running daily linear regression: {e}")

if __name__ == "__main__":
    main()



