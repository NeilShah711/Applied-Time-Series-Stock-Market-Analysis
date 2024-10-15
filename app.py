import streamlit as st
from datetime import datetime
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import talib
from prophet import Prophet
from pygooglenews import GoogleNews
from scipy import stats

# Constants
START = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

st.title("Stock Market Analysis and Prediction")
st.write("Srihari Thyagarajan - I066")
st.write("Avneesh Tilwani - I067")
st.write("Neil Shah - I077")
st.write("Under The Guidance of Prof. Kapil Rathor")
st.write("Github Link: https://github.com/NeilShah711/Applied-Time-Series-Stock-Market-Analysis")

st.write("This app performs stock market analysis and prediction using ARIMA and Prophet models.")
st.write("Each section is explained to help understand why the step was taken.")

def search_stock(symbol):
    st.subheader("Step 1: Search for the Stock")
    st.write(f"Searching for the stock symbol **{symbol}** in Yahoo Finance...")
    try:
        stock = yf.Ticker(symbol)
        if 'longName' in stock.info:
            st.write(f"Stock Found: **{stock.info['longName']}**")
            return stock
        else:
            st.write("No valid stock found. Please check the symbol.")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def load_data(stock):
    st.subheader("Step 2: Load Historical Data")
    st.write(f"Fetching historical data from **{START}** to **{TODAY}**.")
    data = yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    st.write("Loaded Data (Last 5 Rows):")
    st.write(data.tail())
    return data

def apply_technical_indicators(data):
    st.subheader("Step 3: Apply Technical Indicators")
    st.write("Calculating **SMA (Simple Moving Average)** and **RSI (Relative Strength Index)** for analysis.")
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['RSI'] = talib.RSI(data['Close'])
    data.dropna(inplace=True)
    return data

def plot_raw_data(data):
    st.subheader("Step 4: Visualizing Raw Data")
    st.write("This plot shows the **Open** and **Close** prices over time.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(data, column, title):
    st.subheader(f"Step 6: Histogram of {column}")
    st.write(f"A **histogram** displays the distribution of {column} prices.")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[column], name=column))
    fig.update_layout(title_text=title)
    st.plotly_chart(fig)

def plot_qq_plot(data, column):
    st.subheader("Step 7: QQ Plot (Quantile-Quantile Plot)")
    st.write("A **QQ plot** compares the distribution of the data with a normal distribution.")
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(data[column], dist="norm", plot=ax)
    ax.get_lines()[1].set_color('red')
    st.pyplot(fig)

def descriptive_statistics(data):
    st.subheader("Step 5: Descriptive Statistics")
    st.write("This section provides a summary of the stock data with key statistics.")
    st.write(data.describe())

def check_stationarity(data):
    st.subheader("Step 8: Check for Stationarity (ADF Test)")
    
    # Drop NaN values to avoid errors
    data = data.dropna(subset=['Close'])
    
    if data.empty:
        st.error("No valid data available for the ADF test after dropping NaNs.")
        return None  # Exit if no valid data is available

    # Perform ADF test
    result = adfuller(data['Close'])
    
    # Display results
    st.write("**ADF Test Results:**")
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"p-value: {result[1]}")
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write(f"{key}: {value}")

    # Interpret stationarity result
    if result[1] < 0.05:
        st.success("The series is stationary (p-value < 0.05).")
    else:
        st.warning("The series is non-stationary (p-value >= 0.05). Consider applying a transformation.")
    
    return result


def transform_data(data):
    st.subheader("Step 9: Data Transformation")
    st.write("We apply **first-order differencing** to make the data stationary.")
    data['Close'] = data['Close'].diff().dropna()
    return data

def seasonal_decomposition(data):
    st.subheader("Seasonal Decomposition of Time Series")

    # Drop rows with NaN values
    data_clean = data['Close'].dropna()

    # Check if there is sufficient data left after dropping NaNs
    if len(data_clean) < 30:
        st.error("Not enough data points after dropping NaNs for decomposition.")
        return None  # Exit the function if data is insufficient

    # Perform seasonal decomposition (assuming a period of 30 days)
    decomposition = seasonal_decompose(data_clean, model='additive', period=30)

    # Plot the components
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(data_clean, label='Original', color='black')
    axes[0].legend(loc='upper left')
    axes[1].plot(decomposition.trend, label='Trend', color='blue')
    axes[1].legend(loc='upper left')
    axes[2].plot(decomposition.seasonal, label='Seasonal', color='green')
    axes[2].legend(loc='upper left')
    axes[3].plot(decomposition.resid, label='Residual', color='red')
    axes[3].legend(loc='upper left')

    st.pyplot(fig)

def identify_best_arima(data):
    st.subheader("Step 11: Identify the Best ARIMA Model")
    st.write("We try different **ARIMA (Auto-Regressive Integrated Moving Average)** models to find the best fit based on **AIC** and **BIC**.")
    best_aic = float('inf')
    best_order = None
    best_bic = float('inf')
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(data['Close'].dropna(), order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_bic = model_fit.bic
                        best_order = (p, d, q)
                except:
                    continue
    st.write("Best ARIMA Order:", best_order)
    st.write("AIC:", best_aic, " | BIC:", best_bic)
    return best_order, best_aic, best_bic

def fetch_stock_news(symbol):
    st.subheader("Step 12: Fetch Latest Stock News")
    st.write(f"Fetching latest news articles related to **{symbol}**.")
    googlenews = GoogleNews()
    news = googlenews.search(f"{symbol} stock price OR market OR performance", when='1d')
    articles = news['entries']
    st.write(f"Total Articles Found: {len(articles)}")
    return articles

def main():
    st.title("Stock Information")
    symbol = st.text_input("Enter stock symbol:", key="symbol_input")
    n_years = st.slider("Years of prediction:", 1, 5)
    period = n_years * 365

    if symbol:
        stock = search_stock(symbol)
        if stock:
            data = load_data(symbol)
            data = apply_technical_indicators(data)
            plot_raw_data(data)
            descriptive_statistics(data)
            plot_histogram(data, 'Close', 'Histogram of Close Prices')
            plot_qq_plot(data, 'Close')

            if check_stationarity(data)[1] > 0.05:
                data = transform_data(data)
                check_stationarity(data)

            seasonal_decomposition(data)

            st.subheader("Step 13: ACF and PACF Plots")
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            plot_acf(data['Close'].dropna(), ax=ax[0])
            plot_pacf(data['Close'].dropna(), ax=ax[1])
            st.pyplot(fig)

            best_order, _, _ = identify_best_arima(data)
            model = ARIMA(data['Close'].dropna(), order=best_order).fit()
            st.write("ARIMA Model Summary:")
            st.write(model.summary())

            news_articles = fetch_stock_news(symbol)
            for article in news_articles[:5]:
                st.write("Title:", article['title'])
                st.write("Link:", article['link'])

if __name__ == "__main__":
    main()
