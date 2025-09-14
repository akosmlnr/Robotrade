#!/usr/bin/env python3
"""
Debug script to test Polygon API connection and diagnose issues
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta

def test_polygon_api():
    """Test Polygon API connection and diagnose issues"""
    
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    base_url = "https://api.polygon.io"
    
    # Test 1: Market status endpoint
    print("\n🔍 Test 1: Market Status")
    try:
        url = f"{base_url}/v1/marketstatus/now"
        params = {'apikey': api_key}
        
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Market Status Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Symbol validation
    print("\n🔍 Test 2: Symbol Validation (AAPL)")
    try:
        url = f"{base_url}/v3/reference/tickers/AAPL"
        params = {'apikey': api_key}
        
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Symbol Info Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: Aggregates endpoint (the one failing)
    print("\n🔍 Test 3: Aggregates Endpoint (15-minute data)")
    try:
        # Use yesterday's date to avoid weekend issues
        yesterday = datetime.now() - timedelta(days=1)
        start_str = yesterday.strftime('%Y-%m-%d')
        end_str = yesterday.strftime('%Y-%m-%d')
        
        url = f"{base_url}/v2/aggs/ticker/AAPL/range/15/minute/{start_str}/{end_str}"
        params = {
            'apikey': api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        print(f"URL: {url}")
        print(f"Params: {params}")
        
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Aggregates Response Status: {data.get('status', 'Unknown')}")
            print(f"Results Count: {len(data.get('results', []))}")
            if data.get('message'):
                print(f"Message: {data.get('message')}")
            if data.get('results'):
                print(f"Sample Result: {json.dumps(data['results'][0], indent=2)}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 4: Last trade endpoint
    print("\n🔍 Test 4: Last Trade")
    try:
        url = f"{base_url}/v2/last/trade/AAPL"
        params = {'apikey': api_key}
        
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Last Trade Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_polygon_api()
