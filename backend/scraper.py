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
import pytz


POLYGON_API_KEY = '3MF64zqXMDhh62DUOAC0Lduj7MpIXlmC'
DB_LOCATION = 'SQLite_Python.db'

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
    returns = []

    # Extracting only the relevant bits of data from the json object
    ticker = get_stock_quote(symbol, dateA, dateB)

    # You can also take appropriate 
    try :
        symbols = [ticker.get('ticker')] * ticker.get('count')
    except TypeError:
        print('\nE--- Error: You entered a date that has no results, please choose a weekday other than today. ---E\n DateA: {a}\n DateB: {b}\n'.format(b = dateB, a = dateA))
        return None
     
    for quote in ticker.get('results')[:]:
        close.append(quote['c'])
        high.append(quote['h'])
        low.append(quote['l'])
        open.append(quote['o'])
        volume.append((quote['v']-87976017.82703777)//194579)
        timestamp.append((quote['t']//100000))
        returns.append(round(((quote['c'] - quote['o']) / quote['o']) * 100, 3)) # This formula calculates the percentage change between the "open" and "close" prices and expresses it as a percentage return. It represents the gain or loss as a proportion of the opening price.
                        
    stock_quote_dict = {
        'Close' : close,
        'High' : high,
        'Low' : low,
        'Open' : open,
        'Date' : timestamp,
        'Volume' : volume,
        'Symbol' : symbols,
        'Returns' : returns,
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

def createSqliteTable(df, title):
    try:
        sqliteConnection = sqlite3.connect(DB_LOCATION)
        cursor = sqliteConnection.cursor()
        print("C--- Connected to SQLite ---C")

        # Create a new table with the given title if it does not exist
        df.to_sql(title, sqliteConnection, index=False, if_exists='replace')
        sqliteConnection.commit()
        print('C--- Created SQLite table successfully ---C')
        cursor.close()

    except sqlite3.Error as error:
        print("C--- Failed to create sqlite table ---C", error)

    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("C--- The SQLite connection is closed ---C")

def deleteRecord(df, title, symbol):
    try:
        sqliteConnection = sqlite3.connect(database_LOCATION)
        cursor = sqliteConnection.cursor()
        print('D--- Connected to SQLite ---D')

        # Deleting single record now
        sql_delete_query = '''DELETE FROM {title} WHERE Symbol = \'{symbol}\''''.format(symbol=symbol, title=title)
        cursor.execute(sql_delete_query)            
        df.to_sql(title, sqliteConnection, index=False, if_exists='append')
        print("D--- Deleted record successfully ---D")
        sqliteConnection.commit()
        print('D--- Replaced database with a new value for {symbol} successfully ---D'.format(symbol = symbol))
        cursor.close()

    except sqlite3.Error as error:
        print("D--- Failed to delete record from sqlite table ---D", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("D--- The SQLite connection is closed ---D")

def readSqliteTable(title, symbol, date):
    df = pd.DataFrame()
    try:
        # Connect to the SQLite database
        sqliteConnection = sqlite3.connect(DB_LOCATION)
        cursor = sqliteConnection.cursor()
        print("R--- Connected to SQLite ---R")

        # Check if the table with the given title exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{title}'")
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            print(f"R--- Table '{title}' does not exist. Aborting read. ---R")
            return None

        # Build the query based on the provided symbol and date
        query = '''SELECT * FROM {t}'''.format(t = title)
        if symbol and date:
            query = '''SELECT * FROM {t} WHERE Symbol = \'{s}\' AND Date = \'{d}\''''.format(t = title, s = symbol, d = date)
        elif symbol:
            query = '''SELECT * FROM {t} WHERE Symbol = \'{s}\''''.format(t = title, s = symbol)

        # Read data from the table
        df = pd.read_sql_query(query, sqliteConnection)

        cursor.close()
        print('R--- Read data from SQLite table successfully ---R')
        return df

    except sqlite3.Error as error:
        print("R--- Failed to read data from sqlite table ---R", error)
        return df

    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("R--- The SQLite connection is closed ---R")
            return df

def updateSqliteTable(df, title, symbol):
    try:
        sqliteConnection = sqlite3.connect(DB_LOCATION)
        cursor = sqliteConnection.cursor()
        print("U--- Connected to SQLite ---U")
        
        # Check for duplicates before appending
        # merged_df = readSqliteTable(title, None, None)
        # merged_df = pd.concat([merged_df, df], ignore_index=True)
        # print("merged_df", merged_df)
        merged_df = readSqliteTable(title, symbol, None)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        print("merged_df", merged_df)
        merged_df = merged_df.drop_duplicates(subset=['Date'])
        print("merged_df after\n", merged_df)

        merged_df.to_sql(title, sqliteConnection, index=False, if_exists='replace')
        sqliteConnection.commit()
        print('U--- Updated record for {symbol} successfully ---U'.format(symbol=symbol))
        cursor.close()

    except sqlite3.Error as error:
        print("U--- Failed to update sqlite table ---U", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("U--- The SQLite connection is closed ---U")

def get_stock_df(title, symbol, dateA, dateB):
    stock_quotes = get_stock_quote_data(symbol, dateA, dateB)
    stock_quotes_df = pd.DataFrame(data = stock_quotes, columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns'])

    # Validate
    if check_if_valid_data(stock_quotes_df):
        print('<--- The input data is valid and now proceeding to the load stage; --->\n{data}'.format(data = stock_quotes_df))
    else:
        print('E--- ERROR! The input data is invalid and can\'t proceed to the Load stage; ---E\n{df}\n'.format(df = stock_quotes_df))
        return stock_quotes_df

    # Load
    sql_db = readSqliteTable(title, symbol, None)
    print("sql_db",sql_db)
    if not sql_db.empty:
        print("stock_quotes_df", stock_quotes_df)
        updateSqliteTable(stock_quotes_df, title, symbol)
        print('\n<--- Appended new value for {s} in the {t} database from {b} to {a}. --->\n'.format(s = symbol, t = title, b = dateB, a = dateA))
    else:
        createSqliteTable(stock_quotes_df, title)
        print('\n<--- Created a {t} database with {s} data from {b} to {a}. --->\n'.format(s = symbol, t = title, b = dateB, a = dateA))
    print('(Database {t}):'.format(t = title))
    return readSqliteTable(title, symbol, None)        

# Get a list of stock data of the most recent 'weekday' before today.
def prev_weekday(adate, num):
    adate -= datetime.timedelta(days=num)
    while adate.weekday() > 4: # Mon-Fri are 0-4
        adate -= datetime.timedelta(days=1)
    return adate  
   
# if __name__ == '__main__':
#     STOCKS = ['AAPL', 'NKE']
    
#     TODAY = prev_weekday(datetime.datetime.utcnow(), 1)
#     TODAY = TODAY.strftime("%Y-%m-%d")

    # print(readSqliteTable('my_stock_list_quotes', None, None))
#     DATE = ['2022-04-04', '2022-04-08']
    # TITLE = ['my_stock_quotes', 'my_stock_list_quotes']
    # print(get_stock_quote('AAPl', '2023-08-01', '2023-08-02'))
    # print(get_stock_quote_data('BROS', '2023-08-01', '2023-08-02'))
#     print(get_stock_df(TITLE[1], STOCKS[0], DATE[0], DATE[0]))
#     # print(get_stock_df(TITLE[1], STOCKS[1], DATE[0], DATE[0]))
    # print(get_stock_df('my_stock_list_quotes', 'AAPl', '2023-08-02', '2023-08-04'))



