import yfinance as yf
import pandas as pd
import datetime as dt

# Set the ticker symbol and start and end dates
ticker = "STAG"
start_date = dt.datetime.now() - dt.timedelta(days=365*10)
end_date = dt.datetime.now()

# Download the data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Save the data to a CSV file
data.to_csv("STAG.csv")
