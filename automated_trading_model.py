## Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the dataset
file_path = 'XAUUSD_2010--2023.csv' 
data = pd.read_csv(file_path)

# Converting 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Feature Engineering: Add technical indicators
data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
data['EMA_20'] = data['Adj Close'].ewm(span=20, adjust=False).mean()  # 20-day Exponential Moving Average
data['RSI'] = 100 - (100 / (1 + data['Adj Close'].diff(1).apply(lambda x: max(x, 0)).rolling(window=14).mean() / 
                            data['Adj Close'].diff(1).apply(lambda x: -min(x, 0)).rolling(window=14).mean()))
data['Volatility'] = data['High'] - data['Low']  # Daily price range (High - Low)

# Dropping rows with NaN values created by rolling windows
data.dropna(inplace=True)

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Adj Close', 'SMA_20', 'EMA_20', 'RSI', 'Volatility', 'Volume']])

X, y = [], []
sequence_length = 60  # Use the past 60 days for predictions
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])  # Sequence of 60 days
    y.append(scaled_data[i, 0])  # Target: Next day's 'Adj Close'

X, y = np.array(X), np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Build the enhanced LSTM model
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=32),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions and actual values to original scale
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]
actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]

# Calculate metrics
mae = np.mean(np.abs(actual - predictions))  # Mean Absolute Error
mse = np.mean((actual - predictions)**2)    # Mean Squared Error

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Extract test dates
test_dates = data['Date'][-len(actual):]

# Plot the actual vs forecasted prices
plt.figure(figsize=(12, 8))
plt.plot(test_dates, actual, label='Actual Prices', color='blue')
plt.plot(test_dates, predictions, label='Forecasted Prices', color='red')
plt.title('Actual vs Forecasted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Format x-axis for better readability
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# Automate predictions for future prices
def forecast_next_days(last_sequence, num_days, model, scaler):
    last_sequence_scaled = scaler.transform(last_sequence)
    forecast = []
    current_sequence = last_sequence_scaled[-sequence_length:]  # Start with the last sequence

    for _ in range(num_days):
        prediction = model.predict(current_sequence[np.newaxis, :, :])[0, 0]
        forecast.append(prediction)
        
        # Append the prediction and slide the window
        new_entry = np.concatenate(([prediction], current_sequence[-1, 1:]))  # Include other features
        current_sequence = np.vstack((current_sequence[1:], new_entry))

    # Inverse transform the forecasted prices to original scale
    forecast = scaler.inverse_transform(np.concatenate((np.array(forecast).reshape(-1, 1), 
                                                        np.zeros((len(forecast), scaled_data.shape[1]-1))), axis=1))[:, 0]
    return forecast

# Example: Forecast the next 10 days
last_sequence = data.iloc[-sequence_length:][['Adj Close', 'SMA_20', 'EMA_20', 'RSI', 'Volatility', 'Volume']].values
forecasted_prices = forecast_next_days(last_sequence, num_days=10, model=model, scaler=scaler)

print("Forecasted Prices for the next 10 days:", forecasted_prices)
