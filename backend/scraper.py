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
database_LOCATION = 'SQLite_Python.db'

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

def deleteRecord(df, title, symbol):
    try:
        sqliteConnection = sqlite3.connect(database_LOCATION)
        cursor = sqliteConnection.cursor()
        print('D: Connected to SQLite')

        # Deleting single record now
        sql_delete_query = '''DELETE FROM {title} WHERE Symbol = \'{symbol}\''''.format(symbol=symbol, title=title)
        cursor.execute(sql_delete_query)            
        df.to_sql(title, sqliteConnection, index=False, if_exists='append')
        print("D: Deleted record successfully ")
        sqliteConnection.commit()
        print('D: Replaced database with a new value for {symbol} successfully'.format(symbol = symbol))
        cursor.close()

    except sqlite3.Error as error:
        print("D: Failed to delete record from sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("D: The SQLite connection is closed")

def readSqliteTable(title, symbol, date):
    try:
        sqliteConnection = sqlite3.connect('SQLite_Python.db')
        cursor = sqliteConnection.cursor()

        sqlite_select_query = ''
        if date is not None:
            sqlite_select_query = '''SELECT * FROM {title} WHERE Date = {date} AND Symbol = \'{symbol}\''''.format(symbol=symbol, title=title, date=date)
        elif symbol is not None: 
            sqlite_select_query = '''SELECT * FROM {title} WHERE Symbol = \'{symbol}\''''.format(symbol=symbol, title=title)
        else:
            sqlite_select_query = '''SELECT * FROM {title}'''.format(title = title)
        cursor.execute(sqlite_select_query)
        record = cursor.fetchall()
        cursor.close()
        return pd.DataFrame(data = record, columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns'])
    except sqlite3.Error as error:
        print("R: Failed to read data from sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()

def updateSqliteTable(df, title, symbol):
    try:
        sqliteConnection = sqlite3.connect('SQLite_Python.db')
        cursor = sqliteConnection.cursor()
        print("U: Connected to SQLite")

        df.to_sql(title, sqliteConnection, index=False, if_exists='append')
        sqliteConnection.commit()
        print('U: Updated record for {symbol} successfully'.format(symbol = symbol))
        cursor.close()

    except sqlite3.Error as error:
        print("U: Failed to update sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("U: The SQLite connection is closed\n")

def get_stock_df(title, symbol, dateA, dateB):
    stock_quotes = get_stock_quote_data(symbol, dateA, dateB)
    stock_quotes_df = pd.DataFrame(data = stock_quotes, columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns'])

    # Validate
    if check_if_valid_data(stock_quotes_df):
        print('\n- ------- - ------- - ------- - ------- - ------- - ------- -')
        print(' I\P data valid, proceed to load stage:\n{data}'.format(data = stock_quotes_df))
        print('- ------- - ------- - ------- - ------- - ------- - ------- -')
    else:
        print('Data Not valid, can\'t proceed to Load stage\n DF: \n{df}\n'.format(df = stock_quotes_df))
        record = readSqliteTable(title, None, None)
        print('(Database {title}):'.format(title = title))
        return record

    # Load
    if dateA == dateB:
        # Checking for duplicate
        try:
            recordTS = readSqliteTable(title, symbol, None)
            if not recordTS.empty:
                # try: 
                date = float(stock_quotes_df['Date'])
                recordTSD = readSqliteTable(title, symbol, date)
                if not recordTSD.empty:
                    print('This data already exists in the {title} database.\n'.format(title = title))
                    recordT = readSqliteTable(title, None, None)
                    print('(Database {title}):'.format(title = title))
                    return recordT    
                else:
                    updateSqliteTable(stock_quotes_df, title, symbol)  
                    print('\n<--- Appended value for {symbol} in the {title} database with a timestamp of {date}. --->\n'.format(symbol = symbol, title = title, date = date))
            else:
                updateSqliteTable(stock_quotes_df, title, symbol) 
                print('<--- Appended an {symbol} value to database successfully. --->\n'.format(symbol = symbol))
        except AttributeError:
            print('\n<--- New database created and called {title}, having a value for {symbol}. --->\n'.format(title = title, symbol = symbol))
            updateSqliteTable(stock_quotes_df, title, symbol)            
    else: 
        for date in stock_quotes_df['Date']:
            try:
                recordTSD = readSqliteTable(title, symbol, float(date))
                if not recordTSD.empty:
                    print('Data already exists in the database.\n(Failed data I/P):\n{data}\n'.format(data = recordTSD))
                    continue
            except AttributeError:
                updateSqliteTable(stock_quotes_df, title, symbol) 
                print('\n<--- Appended new value for {symbol} in the {title} database with a timestamp of {date}. --->\n'.format(symbol = symbol, title = title, date = date))
                 
    recordT = readSqliteTable(title, None, None)
    print('(Database {title}):'.format(title = title))
    return recordT          
   
# if __name__ == '__main__':
#     STOCKS = ['AAPL', 'NKE']
#     DATE = ['2022-04-04', '2022-04-08']
#     TITLE = ['my_stock_quotes', 'my_stock_list_quotes']
#     # print(get_stock_quote(STOCKS[1], DATE[0], DATE[1]))
#     # print(get_stock_quote_data(STOCKS[1], DATE[0], 'DATE[1]))
#     print(get_stock_df(TITLE[1], STOCKS[0], DATE[0], DATE[0]))
#     # print(get_stock_df(TITLE[1], STOCKS[1], DATE[0], DATE[0]))
#     # print(get_stock_df(TITLE[1], STOCKS[1], DATE[1], DATE[1]))



