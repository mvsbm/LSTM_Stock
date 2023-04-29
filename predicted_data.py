import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('predicted_data.pkl', 'rb') as f:
    predicted_data = pickle.load(f)

# Load STAG and interest rate data
stag_data = pd.read_csv('STAG.csv', parse_dates=['Date'], index_col='Date')
interest_rate_data = pd.read_csv('US10Y.csv', parse_dates=['DATE'], index_col='DATE')

# Merge STAG and interest rate data on index
merged_data = pd.merge(stag_data, interest_rate_data, left_index=True, right_index=True)

# Select only the columns we need
merged_data = merged_data[['Close', 'DGS10']]

# Plot the actual and predicted stock prices from 6 months ago
six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
recent_data = merged_data[merged_data.index >= six_months_ago]

plt.plot(recent_data.index[-30:], recent_data['Close'][-30:], label='Actual')
plt.plot(recent_data.index[-30:], predicted_data, label='Predicted')
plt.legend()
plt.show()
