import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def process_and_train_binary_classification(csv_path: str, test_size: float = 0.3, random_state: int = 42, lookahead_days: int = 7):
    """
    Process the dataset, calculate features, and train a binary classification model to predict price direction.
    :param csv_path: Path to the CSV file containing the dataset.
    :param test_size: Proportion of data to use for testing.
    :param random_state: Random seed for reproducibility.
    :param lookahead_days: Number of days to look ahead for binary target.
    :return: Trained classification model, metrics on train and test sets.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Calculate features
        df['ma_7day'] = df['close'].rolling(7).mean()
        df['ma_14day'] = df['close'].rolling(14).mean()
        df['ma_30day'] = df['close'].rolling(30).mean()

        # RSI Calculation
        df['price_change'] = df['close'].diff()
        df['gain'] = df['price_change'].clip(lower=0)
        df['loss'] = -1 * df['price_change'].clip(upper=0)
        avg_gain = df['gain'].rolling(14).mean()
        avg_loss = df['loss'].rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])

        # Volume-based indicators
        df['volume_ma_7day'] = df['volume'].rolling(7).mean()

        # Create target column (binary: 1 if price goes up, 0 if price goes down)
        df['future_close'] = df['close'].shift(-lookahead_days)  # Shift close price by lookahead_days
        df['target'] = (df['future_close'] > df['close']).astype(int)  # 1 if future_close > current_close

        # Drop rows with null values
        df = df.dropna()

        # Prepare features and target
        feature_columns = [
            'open', 'high', 'low', 'volume',
            'ma_7day', 'ma_14day', 'ma_30day', 'rsi',
            'bb_upper', 'bb_lower', 'bb_middle', 'volume_ma_7day'
        ]
        X = df[feature_columns]
        y = df['target']

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Logistic Regression model
        model = LogisticRegression(random_state=random_state, max_iter=500)
        model.fit(X_train_scaled, y_train)

        # Predict on training and test sets
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)

        # Print Metrics
        print("\nTraining Accuracy:", train_accuracy)
        print("Testing Accuracy:", test_accuracy)
        print("\nPrecision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

        return model, {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise



