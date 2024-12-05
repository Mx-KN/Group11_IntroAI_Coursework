import matplotlib.pyplot as plt

def plot_bollinger_bands(data):
    """
    Plot Close Price and Bollinger Bands.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.plot(data['Date'], data['BB_Upper'], label='Bollinger Upper Band', linestyle='--', color='orange')
    plt.plot(data['Date'], data['BB_Lower'], label='Bollinger Lower Band', linestyle='--', color='green')
    plt.fill_between(data['Date'], data['BB_Lower'], data['BB_Upper'], color='gray', alpha=0.2)
    plt.title('Bitcoin Close Price with Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_rsi(data):
    """
    Plot Relative Strength Index (RSI).
    """
    plt.figure(figsize=(14, 4))
    plt.plot(data['Date'], data['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', color='red', label='Overbought Threshold (70)')
    plt.axhline(30, linestyle='--', color='green', label='Oversold Threshold (30)')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend()
    plt.show()

def plot_macd(data):
    """
    Plot Moving Average Convergence Divergence (MACD).
    """
    plt.figure(figsize=(14, 5))
    plt.plot(data['Date'], data['MACD'], label='MACD', color='red')
    plt.axhline(0, linestyle='--', color='black', label='Zero Line')
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()
    plt.show()
