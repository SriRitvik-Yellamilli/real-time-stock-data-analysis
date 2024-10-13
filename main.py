import json  # JSON library for handling JSON data (not used in current code)
import openai  # OpenAI API for accessing AI functionalities
import pandas as pd  # Data manipulation and analysis library
import streamlit as st  # Streamlit for building web applications
import matplotlib.pyplot as plt  # Plotting library for creating visualizations
import yfinance as yf  # Yahoo Finance API for fetching financial data
import numpy as np  # Numerical operations library
import time  # Time library for handling time-related functions
import os  # Operating system library for file and environment variable management
from dotenv import load_dotenv  # Library for loading environment variables from a .env file
from sklearn.model_selection import train_test_split  # Function for splitting datasets into training and testing sets
from sklearn.linear_model import LinearRegression  # Linear regression model for predictions

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is present; if not, show an error message and stop the application
if openai.api_key is None:
    st.error("Error: OpenAI API key not found. Please set it in your .env file.")
    st.stop()

# Set up the Streamlit page configuration
st.set_page_config(page_title="Real-Time Stock Analysis Tool", page_icon="📈", layout="wide")

# Function to fetch the current stock price for a given ticker symbol
def get_stock_price(ticker):
    try:
        stock_data = yf.Ticker(ticker).history(period=period)  # Get historical stock data
        if stock_data.empty:  # Check if the data is empty
            return "No data available. Ticker might be delisted or invalid."
        return str(stock_data.iloc[-1].Close)  # Return the latest closing price
    except Exception as e:
        return f"Error fetching stock price: {e}"  # Handle any exceptions

# Function to calculate the Simple Moving Average (SMA) for a given ticker symbol and window size
def calculate_SMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period=period).Close  # Get historical closing prices
        return str(data.rolling(window=window).mean().iloc[-1])  # Calculate and return the SMA
    except Exception as e:
        return f"Error calculating SMA: {e}"  # Handle any exceptions

# Function to calculate the Exponential Moving Average (EMA) for a given ticker symbol and window size
def calculate_EMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period=period).Close  # Get historical closing prices
        return str(data.ewm(span=window, adjust=False).mean().iloc[-1])  # Calculate and return the EMA
    except Exception as e:
        return f"Error calculating EMA: {e}"  # Handle any exceptions

# Function to calculate the Relative Strength Index (RSI) for a given ticker symbol
def calculate_RSI(ticker):
    try:
        data = yf.Ticker(ticker).history(period=period).Close  # Get historical closing prices
        delta = data.diff()  # Calculate the difference between consecutive prices
        up = delta.clip(lower=0)  # Clip the positive differences
        down = -1 * delta.clip(upper=0)  # Clip the negative differences
        ema_up = up.ewm(com=14 - 1, adjust=False).mean()  # Calculate EMA of positive differences
        ema_down = down.ewm(com=14 - 1, adjust=False).mean()  # Calculate EMA of negative differences
        rs = ema_up / ema_down  # Calculate the relative strength
        return str(100 - (100 / (1 + rs)).iloc[-1])  # Calculate and return the RSI
    except Exception as e:
        return f"Error calculating RSI: {e}"  # Handle any exceptions

# Function to calculate the Moving Average Convergence Divergence (MACD) for a given ticker symbol
def calculate_MACD(ticker):
    try:
        data = yf.Ticker(ticker).history(period=period).Close  # Get historical closing prices
        shortEMA = data.ewm(span=12, adjust=False).mean()  # Calculate short-term EMA
        longEMA = data.ewm(span=26, adjust=False).mean()  # Calculate long-term EMA
        MACD = shortEMA - longEMA  # Calculate MACD
        signal = MACD.ewm(span=9, adjust=False).mean()  # Calculate signal line
        MACD_histogram = MACD - signal  # Calculate MACD histogram
        return f'{MACD.iloc[-1]:.2f}'  # Return the latest MACD value
    except Exception as e:
        return f"Error calculating MACD: {e}"  # Handle any exceptions

# Function to plot the stock price for a given ticker symbol
def plot_stock_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period=period)  # Get historical stock data
        close_prices = data['Close']  # Get the closing prices
        if close_prices.empty:  # Check if the data is empty
            st.error("No price data found. The stock may be delisted or invalid.")
            return

        price_diff = np.diff(close_prices)  # Calculate the difference in closing prices
        colors = np.where(price_diff > 0, 'green', 'red')  # Determine colors based on price changes

        plt.figure(figsize=(12, 6))  # Set up the plot size
        plt.plot(data.index[:-1], close_prices[:-1], color='blue', label='Close Price')  # Plot closing prices

        # Loop through the prices and color the segments based on price change
        for i in range(1, len(data)):
            color = 'green' if close_prices[i] > close_prices[i - 1] else 'red'
            plt.plot(data.index[i - 1:i + 1], close_prices[i - 1:i + 1], color=color)

        # Fill the areas under the line based on price changes
        plt.fill_between(data.index[:-1], close_prices[:-1], close_prices.min(), where=(price_diff > 0), color='green', alpha=0.3)
        plt.fill_between(data.index[:-1], close_prices[:-1], close_prices.min(), where=(price_diff < 0), color='red', alpha=0.3)

        plt.title(f'{ticker} Stock Price - {period}')  # Set the plot title
        plt.xlabel('Date')  # Set the x-axis label
        plt.ylabel('Price (USD)')  # Set the y-axis label
        plt.grid(True)  # Show the grid
        st.pyplot(plt)  # Display the plot in Streamlit

    except Exception as e:
        st.error(f"Error plotting stock price: {e}")  # Handle any exceptions

# Function to predict the stock price for a given ticker symbol and number of future days
def predict_stock_price(ticker, days=365):
    try:
        data = yf.Ticker(ticker).history(period='1y')  # Get historical stock data for the past year
        data['Date'] = data.index.map(pd.Timestamp.timestamp)  # Convert dates to timestamps for modeling
        X = data['Date'].values.reshape(-1, 1)  # Feature (date) reshaped for model input
        y = data['Close'].values  # Target variable (closing prices)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()  # Create a linear regression model
        model.fit(X_train, y_train)  # Fit the model to the training data

        # Generate future dates for prediction
        future_dates = np.array([pd.Timestamp.today().timestamp() + i * 86400 for i in range(1, days + 1)]).reshape(-1, 1)
        predictions = model.predict(future_dates)  # Make predictions for the future dates

        return predictions[-1]  # Return the predicted price for the last future date
    except Exception as e:
        return f"Error predicting stock price: {e}"  # Handle any exceptions

# Title for the Streamlit app
st.title("📈 Real-Time Stock Data Analysis")

# Sidebar configuration for user input
st.sidebar.title("📊 Stock Analysis Tool")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL")  # Input for stock ticker
period = st.sidebar.selectbox("Select time period:", ["1y", "6mo", "3mo", "1mo", "5d"])  # Dropdown for time period selection
window = st.sidebar.slider("Select SMA/EMA window size:", min_value=5, max_value=50, value=14)  # Slider for window size
predict_days = st.sidebar.number_input("Predict stock price for the next (1-365 days):", min_value=1, max_value=365, value=30)  # Input for prediction days

# Custom CSS for styling the app
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

# Create two columns in the main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Stock Price for {ticker}")  # Subheader for stock price section

    with st.spinner('Fetching real-time stock price...'):  # Show spinner while fetching data
        time.sleep(1)  # Simulate loading time
        stock_price = get_stock_price(ticker)  # Get the current stock price
        st.metric(label="Current Stock Price (USD)", value=stock_price)  # Display the current stock price

    st.subheader("Technical Indicators")  # Subheader for technical indicators section

    with st.spinner('Calculating indicators...'):  # Show spinner while calculating indicators
        col_sma, col_ema, col_rsi, col_macd = st.columns(4)  # Create four columns for indicators
        col_sma.metric(label=f"SMA ({window}-day)", value=calculate_SMA(ticker, window))  # Display SMA
        col_ema.metric(label=f"EMA ({window}-day)", value=calculate_EMA(ticker, window))  # Display EMA
        col_rsi.metric(label="RSI", value=calculate_RSI(ticker))  # Display RSI
        col_macd.metric(label="MACD", value=calculate_MACD(ticker))  # Display MACD

    st.subheader("Stock Price Chart")  # Subheader for stock price chart section

    with st.spinner('Loading chart...'):  # Show spinner while loading chart
        time.sleep(1)  # Simulate loading time
        plot_stock_price(ticker)  # Plot the stock price chart

    st.subheader("Price Prediction")  # Subheader for price prediction section
    with st.spinner('Predicting future stock price...'):  # Show spinner while predicting price
        predicted_price = predict_stock_price(ticker, predict_days)  # Predict future stock price
        st.metric(label=f"Predicted Price in {predict_days} days (USD)", value=f"{predicted_price:.2f}")  # Display predicted price

# Sidebar section for useful links
st.sidebar.markdown("### Useful Links")
st.sidebar.write("[Yahoo Finance](https://finance.yahoo.com/)")  # Link to Yahoo Finance
st.sidebar.write("[Streamlit Documentation](https://docs.streamlit.io/)")  # Link to Streamlit documentation
