import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_linear_regression_classification(file_path, threshold=0.4, sample_size=50):
    """
    Logistic regression classification pipeline for Bitcoin price movement prediction (weekly).
    
    Parameters:
    - file_path: Path to the dataset CSV file.
    - threshold: Decision threshold for classification.
    - sample_size: Number of random dates to sample for prediction.
    """
    print("Starting refined classification pipeline for weekly prediction...\n")

    # 1. Load and clean the data
    print("Loading Bitcoin data...")
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"Data cleaned. Total rows: {len(data)}\n")

    # 2. Add technical indicators with emphasis on 200 MA
    print("Adding moving averages, crossovers, and trend-based features...")
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Distance from 200 MA
    data['Distance_200_MA'] = (data['Close'] - data['SMA_200']) / data['SMA_200']

    # Trend indicators based on 200 MA
    data['Above_200_MA'] = (data['Close'] > data['SMA_200']).astype(int)

    # Crossover features
    data['SMA_50_200_Crossover'] = data['SMA_50'] - data['SMA_200']

    # Momentum relative to 200 MA
    data['Momentum'] = data['Close'] - data['Close'].shift(7)
    data['Momentum_200_MA'] = data['Momentum'] / data['SMA_200']

    data.dropna(inplace=True)
    print(f"Data after adding features: {len(data)} rows\n")

    # Debugging Plot: 200 MA and Distance
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.plot(data['Date'], data['SMA_200'], label='200 MA', color='green')
    plt.title("Close Price with 200 MA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # 3. Create the target column
    print("Creating target column for weekly prediction...")
    data['Target'] = (data['Close'].shift(-7) > data['Close']).astype(int)  # Weekly shift (-7 days)
    data['Close_Shifted'] = data['Close'].shift(-7)  # Weekly shifted close price
    data.dropna(inplace=True)
    print(f"Target distribution:\n{data['Target'].value_counts()}\n")

    # 4. Define features and split the data
    features = data.columns.difference(['Date', 'Close', 'Close_Shifted', 'Target'])
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}\n")

    # 5. Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    print("Model training complete.\n")

    # 7. Evaluate the model
    print("Evaluating the model on test data...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Random sampling for prediction
    print(f"Sampling {sample_size} random dates for prediction...")
    sampled_data = data.sample(n=sample_size, random_state=42)
    sampled_X = scaler.transform(sampled_data[features])
    sampled_y = sampled_data['Target']
    sampled_pred_proba = model.predict_proba(sampled_X)[:, 1]
    sampled_pred = (sampled_pred_proba > threshold).astype(int)

    # Display sampled predictions
    sampled_data['Predicted_Probability'] = sampled_pred_proba
    sampled_data['Predicted_Target'] = sampled_pred
    sampled_data['Actual_Target'] = sampled_y
    print("\nSampled Data Predictions:")
    print(sampled_data[['Date', 'Close', 'Close_Shifted', 'Above_200_MA', 'Predicted_Probability', 'Predicted_Target', 'Actual_Target']])

    # Plot sampled close prices
    plt.figure(figsize=(12, 6))
    plt.plot(sampled_data['Date'], sampled_data['Close'], label='Close Price', marker='o', color='blue')
    plt.plot(sampled_data['Date'], sampled_data['Close_Shifted'], label='Shifted Close (W)', marker='x', color='orange')
    plt.title(f"Random Sampled Close vs. Shifted Close (W)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    print(f"Classification pipeline complete.")
