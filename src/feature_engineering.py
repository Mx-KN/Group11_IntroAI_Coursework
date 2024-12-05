import pandas as pd

def add_technical_indicators(data):
    """
    Add technical indicators to the dataset.
    """
    print("Adding technical indicators...")

    # moving averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # exponential moving average
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

    # relative strength index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # macd
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema

    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['Close'].rolling(window=20).std())
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['Close'].rolling(window=20).std())

    # rate of change (ROC)
    data['ROC'] = data['Close'].pct_change(periods=12) * 100

    print("Technical indicators added.")
    return data
