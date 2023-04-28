import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tqdm import tqdm  # import tqdm for loading status

# Load STAG and interest rate data
stag_data = pd.read_csv('STAG.csv', parse_dates=['Date'], index_col='Date')
interest_rate_data = pd.read_csv('US10Y.csv', parse_dates=['DATE'], index_col='DATE')

# Merge STAG and interest rate data on index
merged_data = pd.merge(stag_data, interest_rate_data, left_index=True, right_index=True)

# Select only the columns we need
merged_data = merged_data[['Close', 'DGS10']]

# Define the time window for the LSTM model
time_window = 180  # 180 days

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(merged_data)

# Create the training data and labels
train_data = scaled_data[:-time_window, :]
train_labels = scaled_data[time_window:, 0]

# Create the test data and labels
test_data = scaled_data[-time_window:, :]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 2)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
for i in tqdm(range(100)):  # add tqdm for loading status
    model.fit(x=train_data.reshape(-1, time_window, 2), y=train_labels, epochs=1, batch_size=32, verbose=0)

# Predict the stock prices for the next 30 days
predicted_data = []
last_window = train_data[-1, :]
for i in tqdm(range(30)):  # add tqdm for loading status
    next_day = model.predict(last_window.reshape(1, time_window, 2))
    predicted_data.append(next_day[0, 0])
    last_window = np.append(last_window[1:, :], next_day, axis=0)

# Inverse transform the predicted data to get the actual stock prices
predicted_data = scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))

# Plot the actual and predicted stock prices for the next 30 days
plt.plot(merged_data.index[-30:], merged_data['Close'][-30:], label='Actual')
plt.plot(merged_data.index[-30:], predicted_data, label='Predicted')
plt.legend()
plt.show()
