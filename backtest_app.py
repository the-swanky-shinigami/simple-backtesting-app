import sys
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
import numpy as np

console = Console()

def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        console.print(f"[red]No data found for {symbol} in the given range.[/red]")
        sys.exit(1)
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
    # Calculate daily returns
    daily_returns = data['Close'].pct_change().dropna()
    # Volatility (annualized std dev)
    volatility = daily_returns.std() * np.sqrt(252) * 100
    if isinstance(volatility, pd.Series):
        volatility = float(volatility.iloc[0])
    else:
        volatility = float(volatility)
    # Sharpe ratio (assume risk-free rate = 0)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    if isinstance(sharpe_ratio, pd.Series):
        sharpe_ratio = float(sharpe_ratio.iloc[0])
    else:
        sharpe_ratio = float(sharpe_ratio)
    # Equity curve
    equity_curve = (1 + daily_returns).cumprod()
    # Max drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100
    if isinstance(max_drawdown, pd.Series):
        max_drawdown = float(max_drawdown.iloc[0])
    else:
        max_drawdown = float(max_drawdown)
    return volatility, sharpe_ratio, max_drawdown

def plot_data(data, symbol, strategy_name, trades=None, trade_log=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # Price chart
    ax1.plot(data['Close'], label='Close Price', color='blue')
    if 'short_ma' in data:
        ax1.plot(data['short_ma'], label='Short MA', color='orange')
    if 'long_ma' in data:
        ax1.plot(data['long_ma'], label='Long MA', color='green')
    # Mark buy/sell points
    if trade_log:
        buy_dates = [pd.to_datetime(trade['date']) for trade in trade_log if trade['action'] == 'BUY']
        buy_prices = [trade['price'] for trade in trade_log if trade['action'] == 'BUY']
        sell_dates = [pd.to_datetime(trade['date']) for trade in trade_log if trade['action'] == 'SELL']
        sell_prices = [trade['price'] for trade in trade_log if trade['action'] == 'SELL']
        ax1.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy', s=100, zorder=5)
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell', s=100, zorder=5)
    ax1.set_title(f'{symbol} - {strategy_name}')
    ax1.set_ylabel('Price')
    ax1.legend()
    # Equity curve
    daily_returns = data['Close'].pct_change().dropna()
    equity_curve = (1 + daily_returns).cumprod()
    ax2.plot(equity_curve.index, equity_curve.values, label='Equity Curve', color='purple')
    ax2.set_ylabel('Equity')
    ax2.set_xlabel('Date')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def print_trade_log(trade_log):
    if not trade_log:
        console.print("[bold yellow]No trades executed.[/bold yellow]")
        return
    table = Table(title="Trade Log")
    table.add_column("Action", justify="center")
    table.add_column("Date", justify="center")
    table.add_column("Price", justify="right")
    for trade in trade_log:
        table.add_row(trade["action"], trade["date"], f"{trade['price']:.2f}")
    console.print(table)

def main():
    if not sys.stdin.isatty():
        print("Please run this script in an interactive console to enter choices.")
        sys.exit(0)
    console.print("[bold green]Welcome to the Simple Backtesting App![/bold green]")
    symbol = Prompt.ask("Enter stock symbol (e.g., AAPL)").upper()
    start = Prompt.ask("Enter start date (YYYY-MM-DD)")
    end = Prompt.ask("Enter end date (YYYY-MM-DD)")
    table = Table(title="Select Strategy")
    table.add_column("Option", justify="center")
    table.add_column("Strategy", justify="left")
    table.add_row("1", "Buy & Hold")
    table.add_row("2", "Moving Average Crossover")
    table.add_row("3", "RSI Strategy")
    table.add_row("4", "Bollinger Bands Strategy")
    console.print(table)
    strategy_choice = Prompt.ask("Enter option number", choices=["1", "2", "3", "4"])
    data = fetch_data(symbol, start, end)
    if strategy_choice == "1":
        returns, trades, trade_log = buy_and_hold(data)
        strategy_name = "Buy & Hold"
    elif strategy_choice == "2":
        short_window = int(Prompt.ask("Enter short moving average window", default="20"))
        long_window = int(Prompt.ask("Enter long moving average window", default="50"))
        returns, trades, trade_log = moving_average_crossover(data, short_window, long_window)
        strategy_name = f"Moving Average Crossover ({short_window}/{long_window})"
    elif strategy_choice == "3":
        rsi_period = int(Prompt.ask("Enter RSI period", default="14"))
        returns, trades, trade_log = rsi_strategy(data, rsi_period)
        strategy_name = f"RSI Strategy (period={rsi_period})"
    else:
        bb_window = int(Prompt.ask("Enter Bollinger Bands window", default="20"))
        bb_std = float(Prompt.ask("Enter number of standard deviations", default="2"))
        returns, trades, trade_log = bollinger_bands_strategy(data, bb_window, bb_std)
        strategy_name = f"Bollinger Bands (window={bb_window}, std={bb_std})"
    volatility, sharpe_ratio, max_drawdown = calculate_performance_metrics(data)
    console.print(f"[cyan]{strategy_name} returns: {returns:.2f}%[/cyan]")
    console.print(f"[yellow]Volatility: {volatility:.2f}%[/yellow]")
    console.print(f"[magenta]Sharpe Ratio: {sharpe_ratio:.2f}[/magenta]")
    console.print(f"[red]Max Drawdown: {max_drawdown:.2f}%[/red]")
    print_trade_log(trade_log)
    plot_data(data, symbol, strategy_name, trades, trade_log)

if __name__ == "__main__":
    main() 