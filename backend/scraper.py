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
database_LOCATION = 'sqlite:///db.sqlite'
conn = sqlite3.connect('db.sqlite')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input('Start date', datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.date(2021, 1, 31))

# Extract: get stock data
def get_stock_quote(symbol, dateA, dateB):
    client = RESTClient(POLYGON_API_KEY) # api_key is used
    quotes = cast(
        HTTPResponse,
        client.get_aggs(
            symbol,
            1,
            'day',
            dateA,
            dateB,
            raw=True,
        ),
    ).data.decode('utf-8')   
    return json.loads(quotes)

# Transform: clean the data
def get_stock_quote_data(symbol, dateA, dateB):
    close = []      
    high = []       
    low = []        
    open = []       
    symbols = []
    timestamp = []
    volume = []     

    # Extracting only the relevant bits of data from the json object 
    ticker = get_stock_quote(symbol, dateA, dateB)
    try :
        symbols = [ticker.get('ticker')] * ticker.get('count')
    except TypeError:
        print('\n<--- Error: You entered a date that has no results, please choose a weekday other than today. --->\n')
        return
    # try:
    #     symbols = [ticker.get('ticker')] * ticker.get('count')
    # except TypeError:
    #     symbols = [ticker.get('ticker')] * 1
    # get only business days 
    # timestamp = pd.date_range(dateA, dateB, freq='B').strftime('%Y-%m-%d').tolist()
            
    for quote in ticker.get('results')[:]:
        close.append(quote['c'])
        high.append(quote['h'])
        low.append(quote['l'])
        open.append(quote['o'])
        volume.append(quote['v'])
        timestamp.append(quote['v'])

                        
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

def get_stock_df(title, sybol, dateA, dateB):
    stock_quotes = get_stock_quote_data(sybol, dateA, dateB)

    # To test the output length of each collumn.
    # print(len(stock_quotes['Symbol']), len(stock_quotes['Date']), len(stock_quotes['Open']), len(stock_quotes['High']), len(stock_quotes['Low']), len(stock_quotes['Close']), len(stock_quotes['Volume']))
    
    stock_quotes_df = pd.DataFrame(data = stock_quotes, columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # stock_quotes_df = stock_quotes_df.transpose()

    # print('Going to Validate...')
    # Validate
    if check_if_valid_data(stock_quotes_df):
        print('Data valid, proceed to Load stage\n')
    else:
        print('Data Not valid, can\'t proceed to Load stage\n DF: \n{df}\n'.format(df = stock_quotes_df))
        return '(Database {title}):\n{db}'.format(title = title, db = pd.read_sql_query('SELECT * FROM {title}'.format(title = title), conn))

    # Load
    for date in stock_quotes_df['Date']:
        try:
            data = pd.read_sql_query('SELECT * FROM {title} where Date = {date}'.format(sybol=sybol, title=title, date=int(date)), conn)
            if not data.empty:
                print('Data already exists in the database.')
                print('(Failed data I/P):\n{data}\n'.format(data = data))
                return '(Database {title}):\n{db}'.format(title = title, db = pd.read_sql_query('SELECT * FROM {title}'.format(title = title), conn))
        except pd.io.sql.DatabaseError:
            print('\n<--- New database called {title} created. --->\n'.format(title = title))

    try:
        stock_quotes_df.to_sql(title, conn, index=False, if_exists='append')
        print('Opened database successfully')
    except:
        print('ERROR: cound not open database')

    conn.commit()
    print('Committed to database successfully')

    return '(Database {title}):\n{db}'.format(title = title, db = pd.read_sql_query('SELECT * FROM {title}'.format(title = title), conn))

if __name__ == '__main__':
    STOCKS = ['AAPL', 'NKE']
    DATE = ['2022-04-04', '2022-04-08']
    TITLE = ['my_stock_quotes', 'my_stock_list_quotes']
    # print(get_stock_quote(STOCKS[1], DATE[0], DATE[1]))
    # print(get_stock_quote_data(STOCKS[1], DATE[0], 'DATE[1]))
    print(get_stock_df(TITLE[1], STOCKS[1], DATE[0], DATE[0]))


