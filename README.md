# **Finance App**

## **Overview**
The Finance App is a web application built using Streamlit, designed to provide users with insights into stock prices, technical indicators, and forecasting using the Prophet algorithm. It also fetches the latest news related to the specified stocks, allowing users to stay informed about market trends.

## **Features**
- Fetch historical stock data from Yahoo Finance.
- Display raw stock data with interactive plots.
- Calculate and visualize technical indicators such as SMA and RSI using TA-Lib.
- Perform time series forecasting with the Prophet algorithm.
- Identify the best ARIMA model for stock price prediction.
- Fetch and display the latest news articles related to the stock market.

## **Technologies Used**
- **Python**: The main programming language used to build the application.
- **Streamlit**: A web framework for creating data applications in Python.
- **Pandas**: For data manipulation and analysis.
- **yFinance**: For fetching stock data from Yahoo Finance.
- **Statsmodels**: For statistical modeling and time series analysis.
- **TA-Lib**: For technical analysis of stock data.
- **Prophet**: For forecasting time series data.
- **pygooglenews**: For fetching the latest news articles related to stocks.
- **Plotly & Matplotlib**: For data visualization.

## **Usage**
1. Enter the stock symbol (e.g., AAPL for Apple Inc.) in the input box.
2. Use the slider to select the number of years for prediction.
3. View the stock's historical data, technical indicators, and forecast.
4. Read the latest news articles related to the stock.

## **Contributing**
Contributions are welcome! If you'd like to contribute to the project, please fork the repository and create a pull request.

## **Acknowledgments**
- Inspired by the need for accessible financial analysis tools.
- Thanks to the creators of the libraries used in this project.
