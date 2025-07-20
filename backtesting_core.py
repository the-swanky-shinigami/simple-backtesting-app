import pandas as pd
import numpy as np

def fetch_data(symbol, start, end):
    import yfinance as yf
    data = yf.download(symbol, start=start, end=end)
    return data

def buy_and_hold(data):
    buy_price = float(data['Close'].iloc[0])
    sell_price = float(data['Close'].iloc[-1])
    buy_date = data.index[0].strftime('%Y-%m-%d')
    sell_date = data.index[-1].strftime('%Y-%m-%d')
    returns = (sell_price - buy_price) / buy_price * 100
    trade_log = [
        {"action": "BUY", "date": buy_date, "price": buy_price},
        {"action": "SELL", "date": sell_date, "price": sell_price}
    ]
    return returns, [buy_price, sell_price], trade_log

def moving_average_crossover(data, short_window=20, long_window=50):
    data = data.copy()
    data['short_ma'] = data['Close'].rolling(window=short_window).mean()
    data['long_ma'] = data['Close'].rolling(window=long_window).mean()
    data['signal'] = 0
    data.loc[data.index[short_window:], 'signal'] = (
        data['short_ma'][short_window:] > data['long_ma'][short_window:]
    ).astype(int)
    data['positions'] = data['signal'].diff()
    buy_signals = data[data['positions'] == 1].index
    sell_signals = data[data['positions'] == -1].index
    trade_log = []
    for idx in buy_signals:
        price = data.loc[idx, 'Close']
        if isinstance(price, pd.Series):
            price = float(price.iloc[0])
        else:
            price = float(price)
        trade_log.append({"action": "BUY", "date": idx.strftime('%Y-%m-%d'), "price": price})
    for idx in sell_signals:
        price = data.loc[idx, 'Close']
        if isinstance(price, pd.Series):
            price = float(price.iloc[0])
        else:
            price = float(price)
        trade_log.append({"action": "SELL", "date": idx.strftime('%Y-%m-%d'), "price": price})
    trade_log = sorted(trade_log, key=lambda x: x['date'])
    if len(buy_signals) == 0 or len(sell_signals) == 0:
        return 0, [], trade_log
    buy_price = data.loc[buy_signals[0], 'Close']
    if isinstance(buy_price, pd.Series):
        buy_price = float(buy_price.iloc[0])
    else:
        buy_price = float(buy_price)
    sell_price = data.loc[sell_signals[-1], 'Close']
    if isinstance(sell_price, pd.Series):
        sell_price = float(sell_price.iloc[0])
    else:
        sell_price = float(sell_price)
    returns = (sell_price - buy_price) / buy_price * 100
    return returns, [buy_price, sell_price], trade_log

def rsi_strategy(data, rsi_period=14):
    data = data.copy()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['signal'] = 0
    data.loc[data['RSI'] < 30, 'signal'] = 1  # Buy
    data.loc[data['RSI'] > 70, 'signal'] = -1 # Sell
    data['positions'] = data['signal'].diff()
    buy_signals = data[data['positions'] == 2].index.tolist()  # from 0 to 1
    sell_signals = data[data['positions'] == -2].index.tolist() # from 0 to -1
    trade_log = []
    for idx in buy_signals:
        price = data.loc[idx, 'Close']
        if isinstance(price, pd.Series):
            price = float(price.iloc[0])
        else:
            price = float(price)
        trade_log.append({"action": "BUY", "date": idx.strftime('%Y-%m-%d'), "price": price})
    for idx in sell_signals:
        price = data.loc[idx, 'Close']
        if isinstance(price, pd.Series):
            price = float(price.iloc[0])
        else:
            price = float(price)
        trade_log.append({"action": "SELL", "date": idx.strftime('%Y-%m-%d'), "price": price})
    trade_log = sorted(trade_log, key=lambda x: x['date'])
    if not buy_signals or not sell_signals:
        return 0, [], trade_log
    buy_price = data.loc[buy_signals[0], 'Close']
    if isinstance(buy_price, pd.Series):
        buy_price = float(buy_price.iloc[0])
    else:
        buy_price = float(buy_price)
    sell_price = data.loc[sell_signals[-1], 'Close']
    if isinstance(sell_price, pd.Series):
        sell_price = float(sell_price.iloc[0])
    else:
        sell_price = float(sell_price)
    returns = (sell_price - buy_price) / buy_price * 100
    return returns, [buy_price, sell_price], trade_log

def bollinger_bands_strategy(data, window=20, num_std=2):
    data = data.copy()
    if len(data) < window:
        return 0, [], []  # Not enough data for rolling window
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper'] = data['MA'] + num_std * data['STD']
    data['Lower'] = data['MA'] - num_std * data['STD']
    required_cols = ['Close', 'Lower', 'Upper']
    if not all(col in data.columns for col in required_cols):
        return 0, [], []
    try:
        data = data.dropna(subset=required_cols)
    except KeyError:
        return 0, [], []
    if data.empty:
        return 0, [], []
    data['signal'] = 0
    data.loc[data['Close'] <= data['Lower'], 'signal'] = 1  # Buy
    data.loc[data['Close'] >= data['Upper'], 'signal'] = -1 # Sell
    data['positions'] = data['signal'].diff()
    buy_signals = data[data['positions'] == 2].index.tolist()
    sell_signals = data[data['positions'] == -2].index.tolist()
    trade_log = []
    for idx in buy_signals:
        price = data.loc[idx, 'Close']
        if isinstance(price, pd.Series):
            price = float(price.iloc[0])
        else:
            price = float(price)
        trade_log.append({"action": "BUY", "date": idx.strftime('%Y-%m-%d'), "price": price})
    for idx in sell_signals:
        price = data.loc[idx, 'Close']
        if isinstance(price, pd.Series):
            price = float(price.iloc[0])
        else:
            price = float(price)
        trade_log.append({"action": "SELL", "date": idx.strftime('%Y-%m-%d'), "price": price})
    trade_log = sorted(trade_log, key=lambda x: x['date'])
    if not buy_signals or not sell_signals:
        return 0, [], trade_log
    buy_price = data.loc[buy_signals[0], 'Close']
    if isinstance(buy_price, pd.Series):
        buy_price = float(buy_price.iloc[0])
    else:
        buy_price = float(buy_price)
    sell_price = data.loc[sell_signals[-1], 'Close']
    if isinstance(sell_price, pd.Series):
        sell_price = float(sell_price.iloc[0])
    else:
        sell_price = float(sell_price)
    returns = (sell_price - buy_price) / buy_price * 100
    return returns, [buy_price, sell_price], trade_log

def calculate_performance_metrics(data):
    daily_returns = data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    if isinstance(volatility, pd.Series):
        volatility = float(volatility.iloc[0])
    else:
        volatility = float(volatility)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    if isinstance(sharpe_ratio, pd.Series):
        sharpe_ratio = float(sharpe_ratio.iloc[0])
    else:
        sharpe_ratio = float(sharpe_ratio)
    equity_curve = (1 + daily_returns).cumprod()
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100
    if isinstance(max_drawdown, pd.Series):
        max_drawdown = float(max_drawdown.iloc[0])
    else:
        max_drawdown = float(max_drawdown)
    return volatility, sharpe_ratio, max_drawdown 