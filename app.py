import streamlit as st
import pandas as pd
import plotly.graph_objs as go
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
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                st.write('Flattening MultiIndex columns...')
                if len(data.columns.levels[1]) == 1:
                    data.columns = data.columns.droplevel(1)
                else:
                    data.columns = ['_'.join(col).strip() for col in data.columns.values]

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
            # --- Ensure datetime index, sorted, and unique ---
            data = data.copy()
            data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep='first')]
            data = data.sort_index()
            # --- Plotly Price Chart ---
            price_fig = go.Figure()
            # Candlestick if OHLC available
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']) and len(data) > 1:
                price_fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                    name='Candlestick',
                    increasing_line_color='green', decreasing_line_color='red',
                    showlegend=True
                ))
            # Always add a close price line for clarity
            if len(data) > 1:
                price_fig.add_trace(go.Scatter(
                    x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue', width=2)
                ))
            # Overlays
            if strategy == "Moving Average Crossover" and len(data) > 1:
                data['short_ma'] = data['Close'].rolling(window=int(short_window)).mean()
                data['long_ma'] = data['Close'].rolling(window=int(long_window)).mean()
                price_fig.add_trace(go.Scatter(x=data.index, y=data['short_ma'], mode='lines', name='Short MA', line=dict(color='orange')))
                price_fig.add_trace(go.Scatter(x=data.index, y=data['long_ma'], mode='lines', name='Long MA', line=dict(color='green')))
            if strategy == "Bollinger Bands" and len(data) > 1:
                data['MA'] = data['Close'].rolling(window=int(bb_window)).mean()
                data['STD'] = data['Close'].rolling(window=int(bb_window)).std()
                data['Upper'] = data['MA'] + float(bb_std) * data['STD']
                data['Lower'] = data['MA'] - float(bb_std) * data['STD']
                price_fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], mode='lines', name='Upper Band', line=dict(color='purple', dash='dash')))
                price_fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], mode='lines', name='Lower Band', line=dict(color='purple', dash='dash')))
            # Buy/Sell markers
            if trade_log and len(trade_log) > 0:
                buy_dates = pd.to_datetime([trade['date'] for trade in trade_log if trade['action'] == 'BUY'])
                buy_prices = [trade['price'] for trade in trade_log if trade['action'] == 'BUY']
                sell_dates = pd.to_datetime([trade['date'] for trade in trade_log if trade['action'] == 'SELL'])
                sell_prices = [trade['price'] for trade in trade_log if trade['action'] == 'SELL']
                if len(buy_dates) > 0:
                    price_fig.add_trace(go.Scatter(
                        x=buy_dates, y=buy_prices, mode='markers', name='Buy',
                        marker=dict(symbol='triangle-up', color='green', size=12)
                    ))
                if len(sell_dates) > 0:
                    price_fig.add_trace(go.Scatter(
                        x=sell_dates, y=sell_prices, mode='markers', name='Sell',
                        marker=dict(symbol='triangle-down', color='red', size=12)
                    ))
            price_fig.update_layout(
                title=f'{symbol} - {strat_name}',
                yaxis_title='Price',
                xaxis_title='Date',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(price_fig, use_container_width=True)
            # --- Plotly Equity Curve ---
            daily_returns = data['Close'].pct_change().dropna()
            equity_curve = (1 + daily_returns).cumprod()
            if len(equity_curve) > 1:
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Equity Curve', line=dict(color='purple', width=2)))
                eq_fig.update_layout(
                    title='Equity Curve',
                    yaxis_title='Equity',
                    xaxis_title='Date',
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(eq_fig, use_container_width=True)
            else:
                st.info('Not enough data to plot equity curve.')
            # --- Plotly RSI (if relevant) ---
            if strategy == "RSI Strategy":
                up = data['Close'].diff().where(data['Close'].diff() > 0, 0).rolling(window=int(rsi_period)).mean()
                down = -data['Close'].diff().where(data['Close'].diff() < 0, 0).rolling(window=int(rsi_period)).mean()
                rs = up / down
                data['RSI'] = 100 - (100 / (1 + rs))
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='brown')))
                rsi_fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')))
                rsi_fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')))
                rsi_fig.update_layout(
                    title='RSI',
                    yaxis_title='RSI',
                    xaxis_title='Date',
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(rsi_fig, use_container_width=True) 