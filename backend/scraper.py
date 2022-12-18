import os

import base64
from contextlib import nullcontext
import copy
from datetime import datetime
import datetime
from distutils.log import error
from flask_cors import CORS, cross_origin
import pandas as pd 
from pyexpat.model import XML_CTYPE_ANY
import requests
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import json
import streamlit as st
import sqlite3
from textwrap import indent
from xml.etree.ElementTree import tostring
from polygon import RESTClient
from typing import cast
from urllib3 import HTTPResponse
from collections import ChainMap


POLYGON_API_KEY = '3MF64zqXMDhh62DUOAC0Lduj7MpIXlmC'
DATABASE_LOCATION = 'sqlite:///db.sqlite'
conn = sqlite3.connect('db.sqlite')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input('Start date', datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.date(2021, 1, 31))


# Extract: get stock data
def get_stock_quote(symbol, dateA, dataB):
    client = RESTClient(POLYGON_API_KEY) # api_key is used
    quotes = cast(
        HTTPResponse,
        client.get_aggs(
            symbol,
            1,
            'day',
            dateA,
            dataB,
            raw=True,
        ),
    ).data.decode('utf-8')   
    return json.loads(quotes)

# Transform: clean the data
def get_stock_quote_data(symbol, dateA, dataB):
    close = []      
    high = []       
    low = []        
    open = []       
    volume = []     

    # Extracting only the relevant bits of data from the json object 
    ticker = get_stock_quote(symbol, dateA, dataB)
    symbols = [ticker.get('ticker')] * ticker.get('count')
    timestamp = pd.date_range(dateA, dataB).strftime('%Y-%m-%d').tolist()
            
    for quote in ticker.get('results')[:]:
        close.append(quote['c'])
        high.append(quote['h'])
        low.append(quote['l'])
        open.append(quote['o'])
        volume.append(quote['v'])
                        
    stock_quote_dict = {
        'Close' : close,
        'High' : high,
        'Low' : low,
        'Open' : open,
        'Date' : timestamp,
        'Volume' : volume,
        'Symbol' : symbols,
    }

    return stock_quote_dict

def check_if_valid_data(df: pd.DataFrame) -> bool:
    # Check if dataframe is empty
    if df.empty:
        print('No stock quotes downloaded. Finishing execution')
        return False 

    # Primary Key Check
    if pd.Series(df['Date']).is_unique:
        pass
    else:
        raise Exception('Primary Key check is violated: At least one of the returned stock quotes has the same date.')

    # Check for nulls
    if df.isnull().values.any():
        print(df.head())
        raise Exception('Null values found')

    return True

def get_stock_df(sybol, dateA, dataB):
    stock_quotes = get_stock_quote_data(sybol, dateA, dataB)
    stock_quotes_df = pd.DataFrame(stock_quotes, columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Validate
    if check_if_valid_data(stock_quotes_df):
        print('Data valid, proceed to Load stage')

    # Load
    print('Opened database successfully')

    try:
        stock_quotes_df.to_sql('my_stock_quotes', conn, index=False, if_exists='replace')
    except:
        print('Data already exists in the database')

    df = pd.read_sql_query('SELECT * FROM my_stock_quotes', conn)

    conn.commit()
    print('Close database successfully')
    return df

if __name__ == '__main__':
    STOCK_NAME = 'AAPL'
#     STOCKS = ['AAPL', 'NKE']
    # print(get_stock_quote(STOCK_NAME, '2022-04-04', '2022-04-08'))
    # print(get_stock_quote_data(STOCK_NAME, '2022-04-04', '2022-04-08'))

    print(get_stock_df(STOCK_NAME, '2022-04-04', '2022-04-08'))

    # print(get_stock_chart_df(STOCK_NAME))
    # print(get_stock_chart(RAPID_API, STOCK_NAME, '10d'))
    



