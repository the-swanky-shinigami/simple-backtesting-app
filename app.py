import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backtesting_core import (
    fetch_data, buy_and_hold, moving_average_crossover, rsi_strategy, bollinger_bands_strategy, calculate_performance_metrics
)

st.set_page_config(page_title="Backtesting App", layout="wide")
st.title("📈 Simple Backtesting App")
st.markdown("""
This app lets you backtest simple trading strategies on any stock using Yahoo Finance data. Select your options below and see the results instantly!
""")

with st.sidebar:
    st.header("Backtest Settings")
    symbol = st.text_input("Stock Symbol", value="AAPL")
    start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end = st.date_input("End Date", pd.to_datetime("today"))
    strategy = st.selectbox(
        "Strategy",
        ["Buy & Hold", "Moving Average Crossover", "RSI Strategy", "Bollinger Bands"]
    )
    st.markdown("---")
    if strategy == "Moving Average Crossover":
        short_window = st.number_input("Short MA Window", min_value=2, max_value=100, value=20)
        long_window = st.number_input("Long MA Window", min_value=2, max_value=200, value=50)
    if strategy == "RSI Strategy":
        rsi_period = st.number_input("RSI Period", min_value=2, max_value=50, value=14)
    if strategy == "Bollinger Bands":
        bb_window = st.number_input("BB Window", min_value=2, max_value=100, value=20)
        bb_std = st.number_input("BB Std Dev", min_value=1.0, max_value=5.0, value=2.0)
    run = st.button("Run Backtest")

if 'run' not in locals():
    run = False

if run:
    with st.spinner("Fetching data and running backtest..."):
        data = fetch_data(symbol, str(start), str(end))
        if data is None or data.empty:
            st.error("No data found for the given symbol and date range.")
        else:
            if strategy == "Buy & Hold":
                returns, trades, trade_log = buy_and_hold(data)
                strat_name = "Buy & Hold"
            elif strategy == "Moving Average Crossover":
                returns, trades, trade_log = moving_average_crossover(data, int(short_window), int(long_window))
                strat_name = f"MA Crossover ({short_window}/{long_window})"
            elif strategy == "RSI Strategy":
                returns, trades, trade_log = rsi_strategy(data, int(rsi_period))
                strat_name = f"RSI (period={rsi_period})"
            else:
                returns, trades, trade_log = bollinger_bands_strategy(data, int(bb_window), float(bb_std))
                strat_name = f"Bollinger Bands (window={bb_window}, std={bb_std})"
            volatility, sharpe_ratio, max_drawdown = calculate_performance_metrics(data)
            st.subheader(f"Results: {strat_name}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Returns (%)", f"{returns:.2f}")
            col2.metric("Volatility (%)", f"{volatility:.2f}")
            col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            col4.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")
            st.markdown("---")
            st.subheader("Trade Log")
            if trade_log:
                st.dataframe(pd.DataFrame(trade_log))
            else:
                st.info("No trades executed.")
            st.markdown("---")
            st.subheader("Charts")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax1.plot(data['Close'], label='Close Price', color='blue')
            if strategy == "Moving Average Crossover":
                data['short_ma'] = data['Close'].rolling(window=int(short_window)).mean()
                data['long_ma'] = data['Close'].rolling(window=int(long_window)).mean()
                ax1.plot(data['short_ma'], label='Short MA', color='orange')
                ax1.plot(data['long_ma'], label='Long MA', color='green')
            if strategy == "Bollinger Bands":
                data['MA'] = data['Close'].rolling(window=int(bb_window)).mean()
                data['STD'] = data['Close'].rolling(window=int(bb_window)).std()
                data['Upper'] = data['MA'] + float(bb_std) * data['STD']
                data['Lower'] = data['MA'] - float(bb_std) * data['STD']
                ax1.plot(data['Upper'], label='Upper Band', color='purple', linestyle='--')
                ax1.plot(data['Lower'], label='Lower Band', color='purple', linestyle='--')
            if strategy == "RSI Strategy":
                up = data['Close'].diff().where(data['Close'].diff() > 0, 0).rolling(window=int(rsi_period)).mean()
                down = -data['Close'].diff().where(data['Close'].diff() < 0, 0).rolling(window=int(rsi_period)).mean()
                rs = up / down
                data['RSI'] = 100 - (100 / (1 + rs))
                ax2.plot(data['RSI'], label='RSI', color='brown')
                ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
                ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
            # Mark buy/sell points
            if trade_log:
                buy_dates = [pd.to_datetime(trade['date']) for trade in trade_log if trade['action'] == 'BUY']
                buy_prices = [trade['price'] for trade in trade_log if trade['action'] == 'BUY']
                sell_dates = [pd.to_datetime(trade['date']) for trade in trade_log if trade['action'] == 'SELL']
                sell_prices = [trade['price'] for trade in trade_log if trade['action'] == 'SELL']
                ax1.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy', s=100, zorder=5)
                ax1.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell', s=100, zorder=5)
            ax1.set_title(f'{symbol} - {strat_name}')
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
            st.pyplot(fig) 