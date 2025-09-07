# ==============================================================================

import pandas as pd
import yfinance as yf
import datetime

# show data for different tickers
start = pd.to_datetime('2004-08-01')
stock = ['ETH-USD']
data = yf.download(stock, start=start, end=datetime.date.today())
print(data)

stock = ['GOOG']
data = yf.download(stock, start=start, end=datetime.date.today())
print(data)

stock = ['FB']
data = yf.download(stock, start=start, end=datetime.date.today())
print(data)

stock = ['TSLA']
data = yf.download(stock, start=start, end=datetime.date.today())
print(data)
