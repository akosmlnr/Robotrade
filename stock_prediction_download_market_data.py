# ==============================================================================

import pandas as pd
from polygon_data_fetcher import download
import datetime

# show data for different tickers
start = pd.to_datetime('2004-08-01')
api_key = "zjBfWbPFc5gE5AgPZpmghkcVkuak0azA"

# Note: ETH-USD is a crypto ticker, using ETH instead for Polygon.io
stock = ['ETH']
data = download(stock, start=start, end=datetime.date.today(), api_key=api_key)
print(data)

stock = ['GOOGL']  # Google's ticker on Polygon.io
data = download(stock, start=start, end=datetime.date.today(), api_key=api_key)
print(data)

stock = ['META']  # Facebook is now Meta
data = download(stock, start=start, end=datetime.date.today(), api_key=api_key)
print(data)

stock = ['TSLA']
data = download(stock, start=start, end=datetime.date.today(), api_key=api_key)
print(data)
