import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)
import numpy as np


def run_daily_linear_regression(file_path, threshold=0.6):
    """
    Run the daily linear regression pipeline to classify Bitcoin price movements.
    
    Parameters:
    - file_path: Path to the dataset CSV file.
    - threshold: Decision threshold for classification (default=0.6).
    """
    print("Starting daily linear regression pipeline...")

    # Load the dataset
    print("Loading Bitcoin data...")
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Clean the data
    print("Cleaning Bitcoin data...")
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convert Date to datetime
    data.dropna(subset=['Date'], inplace=True)  # Remove rows with invalid dates
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Keep relevant columns
    print(f"Data cleaned. Total rows: {len(data)}")

    # Add technical indicators
    print("Adding technical indicators...")
    
    # Moving averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # Crossovers
    data['SMA_10_50_Crossover'] = data['SMA_10'] - data['SMA_50']
    data['SMA_50_200_Crossover'] = data['SMA_50'] - data['SMA_200']
    data['EMA_10_50_Crossover'] = data['EMA_10'] - data['EMA_50']

    # Trend-based features
    data['Trend_SMA_10'] = data['Close'] / data['SMA_10']
    data['Trend_SMA_50'] = data['Close'] / data['SMA_50']
    data['Trend_SMA_200'] = data['Close'] / data['SMA_200']
    
    # Momentum and volatility
    data['Momentum'] = data['Close'] - data['Close'].shift(1)
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']
    
    # Drop rows with insufficient data for indicators
    data.dropna(inplace=True)
    print(f"Data after adding technical indicators: {len(data)} rows")

    # Debug plot: Where the price sits relative to SMA_200
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Trend_SMA_200'], label='Close / SMA_200', color='orange')
    plt.axhline(1, linestyle='--', color='red', label='Neutral Zone (1)')
    plt.title("Price Position Relative to SMA_200")
    plt.xlabel("Date")
    plt.ylabel("Relative Value")
    plt.legend()
    plt.show()

    # Create binary target column
    print("Creating binary target column...")
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price increases
    data.dropna(inplace=True)  # Drop rows with invalid target
    print(f"Target column created. Target distribution:\n{data['Target'].value_counts()}")

    # Define features and split the data
    features = data.columns.difference(['Date', 'Close', 'Target'])
    print(f"Splitting data using features: {list(features)}...")
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train Logistic Regression model
    print("Training Logistic Regression model with balanced class weighting...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Debugging plot: Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Decrease", "Increase"])
    plt.title("Confusion Matrix")
    plt.show()

    # Feature correlation
    print("\nFeature Correlation with Target:")
    print(data.corr()['Target'].sort_values(ascending=False))
    print("\nDaily linear regression pipeline complete.")
