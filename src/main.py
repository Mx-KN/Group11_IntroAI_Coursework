from pathlib import Path
from data_preprocessing import load_data, clean_data
from feature_engineering import add_technical_indicators
from linear_model_training import run_classification
from plot import plot_bollinger_bands, plot_macd, plot_rsi

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

    # # Step 4: Display the final DataFrame
    # print("\nFinal DataFrame (First 5 Rows):")
    # print(final_data.head())
    # print("\nSummary Statistics:")
    # print(final_data.describe())

    # # Step 5: Plot technical indicators
    # plot_bollinger_bands(final_data)
    # plot_rsi(final_data)
    # plot_macd(final_data)

    print("\nRunning classification...")
    # run_classification(final_data)


#            0       0.00      0.00      0.00       363
#            1       0.51      1.00      0.67       374

#     accuracy                           0.51       737
#    macro avg       0.25      0.50      0.34       737
# weighted avg       0.26      0.51      0.34       737
# Target
# 1    1951
# 0    1731

# 1. Class Imbalance: The dataset has a slight imbalance (53% for class 1 vs 47% for class 0), 
#    which can cause the model to favor the majority class (class 1) and fail to predict the minority class.
# 2. Model Bias: Logistic Regression is a simple linear model that tends to perform poorly on imbalanced data 
#    without proper handling, leading to poor precision and recall for the minority class.
# 3. Predictive Power: The technical indicators may not provide sufficient discriminatory power. 
#    Improving feature engineering or using more advanced models can enhance performance.
#  Solution:
# 1. Handle Class Imbalance:
#    - Use class weighting (class_weight='balanced') in Logistic Regression to account for imbalance.
#    - Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the training data by oversampling
#      the minority class.
#
# 2. Adjust Decision Threshold:
#    - Instead of using the default threshold (0.5), tune the threshold to optimize precision and recall
#      for the project’s goals.
#
# 3. Improve Evaluation:
#    - Use metrics like precision, recall, and F1-score to assess the model's performance for both classes.
#    - A classification report will give a more detailed view of how well the model is performing.
# 

    run_classification(final_data, use_smote=True, threshold=0.6) 
    print(final_data['Target'].value_counts())



if __name__ == "__main__":
    main()
