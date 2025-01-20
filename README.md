# Automated-Trading-Model

The automated trading model leverages machine learning and time-series analysis to forecast stock prices. It uses historical data as input and produces future price predictions, helping assess potential trends in the stock market.
This project demonstrates:
* Predictive Modeling: Utilizing an LSTM (Long Short-Term Memory) neural network for forecasting.
* Technical Indicators: Incorporating moving averages (SMA, EMA), RSI (Relative Strength Index), and volatility measures.
* Data Visualization: Comparing actual prices with predicted prices to evaluate model performance.

## How It Works
Steps:
1. Data Preprocessing:
    * Load and clean historical stock data.
    * Add technical indicators such as SMA, EMA, RSI, and volatility.
    * Normalize the data for compatibility with the LSTM neural network.
2. Model Training:
    * Split data into training and testing sets.
    * Train an LSTM model to predict stock prices based on sequences of historical data.
3. Price Prediction:
    * Forecast stock prices for the test set.
    * Use the trained model to predict prices for the next 10 days.
4. Visualization:
    * Plot and compare actual vs. predicted prices for easy interpretation.
Example Dataset Used:
* The program was tested using Tesla (TSLA) stock data to predict future stock prices.

## Results
The model produces a plot comparing actual and forecasted prices. Below are two sample outputs from the project:
Visualization:
* TSLA Example: The program generated the following results using historical TSLA data:
    * Blue Line: Actual prices.
    * Red Line: Predicted prices by the model.


![Automated TSLA](https://github.com/user-attachments/assets/c76b8897-6169-4056-a20c-95fe37085eb5)





## Key Features
* Machine Learning Model: Implements a custom LSTM architecture.
* Technical Analysis: Enhances predictions using indicators like SMA, EMA, and RSI.
* Visual Results: Clear plots to showcase model performance.

