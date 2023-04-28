import pandas_datareader as pdr
import datetime as dt

start_date = dt.datetime.now() - dt.timedelta(days=365 * 10) # 10 years ago
end_date = dt.datetime.now()

df = pdr.DataReader('DGS10', 'fred', start_date, end_date)
df.to_csv('US10Y.csv')
