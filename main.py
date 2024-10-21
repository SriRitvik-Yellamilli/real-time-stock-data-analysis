import json
import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import time
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    st.error("Error: OpenAI API key not found. Please set it in your .env file.")
    st.stop()

st.set_page_config(page_title="Real-Time Stock Analysis Tool", page_icon="📈", layout="wide")

period = '1y'

def convert_ticker(ticker):
    return ticker.replace(':', '-').replace('.', '-')

def is_valid_ticker(ticker):
    try:
        ticker = convert_ticker(ticker)
        stock_data = yf.Ticker(ticker).history(period='1d')
        return not stock_data.empty
    except Exception:
        return False

def get_stock_price(ticker):
    try:
        ticker = convert_ticker(ticker)
        stock_data = yf.Ticker(ticker).history(period=period)
        if stock_data.empty:
            return "No data available. Ticker might be delisted or invalid."
        return str(stock_data.iloc[-1].Close)
    except Exception as e:
        return f"Error fetching stock price: {e}"

def calculate_SMA(ticker, window):
    try:
        ticker = convert_ticker(ticker)
        data = yf.Ticker(ticker).history(period=period).Close
        return str(data.rolling(window=window).mean().iloc[-1])
    except Exception as e:
        return f"Error calculating SMA: {e}"

def calculate_EMA(ticker, window):
    try:
        ticker = convert_ticker(ticker)
        data = yf.Ticker(ticker).history(period=period).Close
        return str(data.ewm(span=window, adjust=False).mean().iloc[-1])
    except Exception as e:
        return f"Error calculating EMA: {e}"

def calculate_RSI(ticker):
    try:
        ticker = convert_ticker(ticker)
        data = yf.Ticker(ticker).history(period=period).Close
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=14 - 1, adjust=False).mean()
        ema_down = down.ewm(com=14 - 1, adjust=False).mean()
        rs = ema_up / ema_down
        return str(100 - (100 / (1 + rs)).iloc[-1])
    except Exception as e:
        return f"Error calculating RSI: {e}"

def calculate_MACD(ticker):
    try:
        ticker = convert_ticker(ticker)
        data = yf.Ticker(ticker).history(period=period).Close
        shortEMA = data.ewm(span=12, adjust=False).mean()
        longEMA = data.ewm(span=26, adjust=False).mean()
        MACD = shortEMA - longEMA
        signal = MACD.ewm(span=9, adjust=False).mean()
        MACD_histogram = MACD - signal
        return f'{MACD.iloc[-1]:.2f}'
    except Exception as e:
        return f"Error calculating MACD: {e}"

def plot_stock_price(ticker):
    try:
        ticker = convert_ticker(ticker)
        data = yf.Ticker(ticker).history(period=period)
        close_prices = data['Close'].dropna()
        if close_prices.empty:
            st.error("No price data found. The stock may be delisted or invalid.")
            return

        price_diff = np.diff(close_prices)
        colors = np.where(price_diff > 0, 'green', 'red')

        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-1], close_prices[:-1], color='blue', label='Close Price')

        for i in range(1, len(data)):
            color = 'green' if close_prices[i] > close_prices[i - 1] else 'red'
            plt.plot(data.index[i - 1:i + 1], close_prices[i - 1:i + 1], color=color)

        plt.fill_between(data.index[:-1], close_prices[:-1], close_prices.min(), where=(price_diff > 0), color='green', alpha=0.3)
        plt.fill_between(data.index[:-1], close_prices[:-1], close_prices.min(), where=(price_diff < 0), color='red', alpha=0.3)

        plt.title(f'{ticker} Stock Price - {period}')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error plotting stock price: {e}")

def predict_stock_price(ticker, days=365):
    try:
        ticker = convert_ticker(ticker)
        data = yf.Ticker(ticker).history(period='1y')
        data['Date'] = data.index.map(pd.Timestamp.timestamp)
        X = data['Date'].values.reshape(-1, 1)
        y = data['Close'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        future_dates = np.array([pd.Timestamp.today().timestamp() + i * 86400 for i in range(1, days + 1)]).reshape(-1, 1)
        predictions = model.predict(future_dates)

        return predictions[-1]
    except Exception as e:
        return f"Error predicting stock price: {e}"

st.title("📈 Real-Time Stock Data Analysis")

st.sidebar.title("📊 Stock Analysis Tool")
ticker = st.sidebar.text_input("Enter stock ticker (ONLY UPPERCASE LETTERS):", value="AAPL")
period = st.sidebar.selectbox("Select time period:", ["1y", "6mo", "3mo", "1mo", "5d"])
window = st.sidebar.slider("Select SMA/EMA window size:", min_value=5, max_value=50, value=14)
predict_days = st.sidebar.number_input("Predict stock price for the next (1-365 days):", min_value=1, max_value=365, value=30)

if ticker and not is_valid_ticker(ticker):
    st.error("Invalid ticker symbol. Please enter a valid stock ticker.")
else:
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color:#fff;
            transition: background-color 0.3s ease;
            border-radius: 10px;
        }
        div.stButton > button:first-child:hover {
            background-color: #45a049;
            transform: scale(1.05);
            cursor: pointer;
        }
        div.stTextInput > div > input {
            border-radius: 5px;
            transition: border 0.3s ease;
        }
        div.stTextInput > div > input:focus {
            border-color: #4CAF50;
            box-shadow: 0px 0px 5px rgba(76, 175, 80, 0.5);
        }
        </style>
        """, unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Stock Price for {ticker}")

        with st.spinner('Fetching real-time stock price...'):
            time.sleep(1)
            stock_price = get_stock_price(ticker)
            st.metric(label="Current Stock Price (USD)", value=stock_price)

        st.subheader("Technical Indicators")

        with st.spinner('Calculating indicators...'):
            col_sma, col_ema, col_rsi, col_macd = st.columns(4)
            col_sma.metric(label=f"SMA ({window}-day)", value=calculate_SMA(ticker, window))
            col_ema.metric(label=f"EMA ({window}-day)", value=calculate_EMA(ticker, window))
            col_rsi.metric(label="RSI", value=calculate_RSI(ticker))
            col_macd.metric(label="MACD", value=calculate_MACD(ticker))

        st.subheader("Stock Price Chart")

        with st.spinner('Loading chart...'):
            time.sleep(1)
            plot_stock_price(ticker)

        st.subheader("Price Prediction")
        with st.spinner('Predicting future stock price...'):
            predicted_price = predict_stock_price(ticker, predict_days)
            if isinstance(predicted_price, float):
                st.metric(label=f"Predicted Price in {predict_days} days (USD)", value=f"{predicted_price:.2f}")
            else:
                st.error(predicted_price)

    st.sidebar.markdown("### Useful Links")
    st.sidebar.write("[Yahoo Finance](https://finance.yahoo.com/)")
    st.sidebar.write("[Streamlit Documentation](https://docs.streamlit.io/)")