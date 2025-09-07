# ==============================================================================

import json
from polygon_data_fetcher import PolygonTicker

api_key = "zjBfWbPFc5gE5AgPZpmghkcVkuak0azA"
sec = PolygonTicker("FTSE", api_key)  # Note: FTSE index might not be available, using FTSE as example

data = sec.history()
#data.head()

if not data.empty:
    my_max = data['Close'].idxmax()
    my_min = data['Close'].idxmin()
    print(f"Max Close: {data['Close'].max()} at {my_max}")
    print(f"Min Close: {data['Close'].min()} at {my_min}")

print('Info')
print(json.dumps(sec.info, indent=4, sort_keys=True))
print()

# Note: Polygon.io provides different data structure than Yahoo Finance
# Some of the detailed financial data might not be available through the basic API
print('Note: Polygon.io provides different data structure than Yahoo Finance.')
print('Some detailed financial data (dividends, splits, earnings, etc.) may require')
print('additional API calls or premium subscription.')
print()

# Available data from Polygon.io
print('Available data from Polygon.io:')
print('- Basic ticker information')
print('- Historical price data (OHLCV)')
print('- Current price data')
print('- Company details (name, market, exchange, etc.)')
print()

# Example of getting more detailed info if available
if sec.info:
    print('Company Details:')
    for key, value in sec.info.items():
        if value:  # Only print non-empty values
            print(f'{key}: {value}')