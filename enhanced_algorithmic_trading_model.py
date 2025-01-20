# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Loading dataset
file_path = 'SP_2000.csv'  
data = pd.read_csv(file_path)


# Ensure the necessary columns exist
print(data.columns)

# Select relevant columns for processing
prices = data[['Adj Close']]  

# Feature Engineering: Add Moving Averages and Volatility 
prices['SMA_20'] = prices['Adj Close'].rolling(window=20).mean()
prices['EMA_20'] = prices['Adj Close'].ewm(span=20, adjust=False).mean()
prices['Volatility'] = data['High'] - data['Low'] 

# Drop rows with NaN values after feature engineering
prices.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Split the data into training and testing sets
X = []
y = []

# Use a sequence length of 60 days for LSTM input
sequence_length = 60
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])  # Past 60 days
    y.append(scaled_data[i, 0])  # Target: next day's price (adjusted close)

X, y = np.array(X), np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse transform to get actual prices
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]
actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1]-1))), axis=1))[:, 0]

# Fix the offset
offset = actual.mean() - predictions.mean()
adjusted_predictions = predictions + offset

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Prices', color='blue')
plt.plot(predictions, label='Forecasted Prices', color='orange')
#plt.plot(adjusted_predictions, label='Adjusted Forecasted Prices', color='green')
plt.title('Actual vs Forecasted Prices')
plt.legend()
plt.show()

# Evaluate the model performance
mse = np.mean((actual - adjusted_predictions)**2)
print(f"Mean Squared Error: {mse}")
