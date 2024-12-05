
---

## Technical Indicators: Bollinger Bands, Moving Averages, and RSI

This document provides detailed explanations, formulas, and Python implementations for calculating **Bollinger Bands**, **Moving Averages (SMA/EMA)**, and **Relative Strength Index (RSI)**.

---

### **1. Bollinger Bands (BB)**

#### **What Are Bollinger Bands?**
- A volatility indicator that uses a moving average (MA) and standard deviation to create upper and lower bands.
- Helps identify overbought or oversold conditions and potential price breakouts.

#### **Formula:**
- **Middle Band (BB_Mid):** Simple Moving Average (SMA) of the closing price.
- **Upper Band (BB_Upper):** `BB_Mid + (k * Standard Deviation)`
- **Lower Band (BB_Lower):** `BB_Mid - (k * Standard Deviation)`

Where:
- `k` is typically set to 2 for a 95% confidence interval.

#### **Implementation in Python:**
```python
def add_bollinger_bands(data, window=20):
    data['BB_Mid'] = data['Close'].rolling(window=window).mean()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['Close'].rolling(window=window).std())
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['Close'].rolling(window=window).std())
    return data
```

---

### **2. Moving Averages (MA)**

#### **What Are Moving Averages?**
- A smoothing indicator that helps track price trends by averaging the closing prices over a specific period.

#### **Types:**
1. **Simple Moving Average (SMA):**
   - Equal weight to all prices in the window.
   - Formula:  
     `SMA = (P1 + P2 + ... + Pn) / n`  
     Where `n` is the window size (e.g., 10 or 50 days).

2. **Exponential Moving Average (EMA):**
   - Assigns higher weight to recent prices.
   - Formula:  
     `EMA_t = (P_t * α) + EMA_{t-1} * (1 - α)`  
     Where:  
     - `P_t` is the current price.  
     - `α = 2 / (n + 1)` is the smoothing factor.  
     - `EMA_{t-1}` is the EMA of the previous day.

#### **Implementation in Python:**
```python
def add_moving_averages(data, short_window=10, long_window=50):
    data['SMA_10'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_50'] = data['Close'].rolling(window=long_window).mean()
    data['EMA_10'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    return data
```

---

### **3. Relative Strength Index (RSI)**

#### **What Is RSI?**
- A momentum oscillator that measures the speed and change of price movements.
- Indicates overbought (RSI > 70) or oversold (RSI < 30) conditions.

#### **Formula:**
1. Calculate price changes:  
   `ΔP = P_t - P_{t-1}`  

2. Separate gains and losses:  
   - `Gain = max(ΔP, 0)`  
   - `Loss = max(-ΔP, 0)`  

3. Calculate average gain and loss over a window (`n`):  
   - `AvgGain = Sum of Gains / n`  
   - `AvgLoss = Sum of Losses / n`  

4. Compute relative strength:  
   - `RS = AvgGain / AvgLoss`  

5. Compute RSI:  
   - `RSI = 100 - (100 / (1 + RS))`

#### **Implementation in Python:**
```python
def add_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data
```

---

### **Summary of Indicators**

| **Indicator**       | **Formula**                                                    | **Purpose**                                    |
|----------------------|----------------------------------------------------------------|------------------------------------------------|
| **Bollinger Bands**  | Mid: SMA, Upper: SMA + (2 * StdDev), Lower: SMA - (2 * StdDev) | Measures volatility and price breakout levels. |
| **SMA/EMA**          | SMA: Mean of prices, EMA: Weighted mean emphasizing recent prices | Tracks trends over time.                      |
| **RSI**              | RSI = 100 - (100 / (1 + RS)), where RS = AvgGain / AvgLoss     | Indicates overbought/oversold conditions.      |

---
