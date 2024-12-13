import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

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
    data['Date'] = pd.to_datetime(data['Date'])  # Convert Date to datetime
    data.dropna(inplace=True)  # Drop missing values
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Keep relevant columns
    print("Data cleaning complete.")

    # Add technical indicators
    print("Adding technical indicators...")
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema

    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['Close'].rolling(window=20).std())
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['Close'].rolling(window=20).std())

    print("Technical indicators added.")

    # Drop rows with insufficient data for indicators
    data.dropna(inplace=True)

    # Create binary target column
    print("Creating binary target column...")
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)  # Drop rows with invalid target
    print("Binary target column created.")

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
    print(f"Using decision threshold: {threshold}")
    y_pred = (y_pred_proba > threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(data['Target'].value_counts())
    print("Feature Correlation with Target:")
    print(data.corr()['Target'])



    print("Daily linear regression pipeline complete.")
