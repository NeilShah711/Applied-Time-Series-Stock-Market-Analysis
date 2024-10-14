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
import requests
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

def search_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        if 'longName' in stock.info:
            return stock
        else:
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def load_data(stock):
    data = yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

def apply_technical_indicators(data):
    # Calculate technical indicators using TA-Lib
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['RSI'] = talib.RSI(data['Close'])
    data.dropna(inplace=True)  
    return data

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(data, column, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[column], name=column))
    fig.update_layout(title_text=title)
    st.plotly_chart(fig)

def plot_qq_plot(data, column):
    st.subheader("QQ Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(data[column], dist="norm", plot=ax)
    ax.get_lines()[1].set_color('red')  
    st.pyplot(fig)

def descriptive_statistics(data):
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

def check_stationarity(data):
    result = adfuller(data['Close'])
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:', result[4])
    return result

def transform_data(data):
    # Here I have used first order differncing on the close prices
    data['Close'] = data['Close'].diff().dropna()
    return data

def seasonal_decomposition(data):
    decomposition = seasonal_decompose(data['Close'], model='additive', period=30)
    st.subheader("Seasonal Decomposition")
    fig = decomposition.plot()
    st.pyplot(fig)

def identify_best_arima(data):
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
    return best_order, best_aic, best_bic

def fetch_stock_news(symbol):
    googlenews = GoogleNews()
    news = googlenews.search(f"{symbol} stock price OR market OR performance OR earnings OR financial OR news", when='1d')
    articles = news['entries']
    st.write(f"Total Number of Articles: {len(articles)}")
    return articles

def main():
    st.title("Stock Information")
    symbol = st.text_input("Enter stock symbol:", key="symbol_input")
    n_years = st.slider("Years of prediction:", 1, 5)
    period = n_years * 365
    
    if symbol:
        if 'last_symbol' not in st.session_state:
            st.session_state.last_symbol = ""
        if symbol != st.session_state.last_symbol:
            st.session_state.last_symbol = symbol
            st.empty()  # Clear previous plots
            
        stock = search_stock(symbol)
        if stock:
            st.write("Stock found:")
            data_load_state = st.text("Loading data...")
            data = load_data(symbol)
            data_load_state.text("Data Loaded Successfully!")
            st.subheader("Raw Data")
            st.write(data.tail())
            
            # Prophet Forecasting
            df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            model_prophet = Prophet()
            model_prophet.fit(df_prophet)
            future = model_prophet.make_future_dataframe(periods=period)
            forecast = model_prophet.predict(future)

            # Plot Prophet Forecast
            st.subheader("Prophet Forecast")
            fig_prophet = go.Figure()
            fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Historical Close'))
            fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecasted Close'))
            fig_prophet.layout.update(title_text="Prophet Forecast", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_prophet, use_container_width=True)

            # Apply technical indicators
            data = apply_technical_indicators(data)

            # Plot Raw Data
            plot_raw_data(data)
            
            # Descriptive Statistics
            descriptive_statistics(data)

            # Histogram
            plot_histogram(data, 'Close', 'Histogram of Close Prices')

            #QQ Plot
            plot_qq_plot(data, 'Close')

            # Check Stationarity
            adf_result = check_stationarity(data)
            if adf_result[1] > 0.05:  # p-value > 0.05 means non-stationary
                st.write("Data is non-stationary, applying transformation.")
                data = transform_data(data).dropna()  # Drop NaN after transformation
                check_stationarity(data)  # Check again after transformation

            # Seasonal Decomposition
            seasonal_decomposition(data)

            # ACF and PACF plots
            st.subheader("ACF and PACF Plots")
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            plot_acf(data['Close'].dropna(), ax=ax[0])
            plot_pacf(data['Close'].dropna(), ax=ax[1])
            st.pyplot(fig)

            # Identify best ARIMA model
            best_order, best_aic, best_bic = identify_best_arima(data)
            st.write("Best ARIMA Order:", best_order)
            st.write("AIC:", best_aic)
            st.write("BIC:", best_bic)

            # Fit ARIMA model
            model = ARIMA(data['Close'].dropna(), order=best_order)  # Use the best order
            model_fit = model.fit()
            st.write("Model Summary:")
            st.write(model_fit.summary())

            # Fetch and display stock news
            st.subheader("Latest Stock News")
            news_articles = fetch_stock_news(symbol)
            if news_articles:
                for article in news_articles[:5]:  # Display top 5 articles
                    st.write("Title:", article['title'])
                    st.write("Link:", article['link'])
                    st.write("Published:", article['published'])
            else:
                st.write("No news articles found.")

        else:
            st.write("Stock not found.")

if __name__ == "__main__":
    main()
