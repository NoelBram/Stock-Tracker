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

IEX_API = "pk_913ba7d52f144907a92856b52ea0636e"
DATABASE_LOCATION = "sqlite:///db.sqlite"
conn = sqlite3.connect('db.sqlite')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

def getStockChart(token, s, r):
    endpoint = "https://cloud.iexapis.com/stable/stock/{symbol}/chart/{range}?token={token}&chartByDay=true".format(symbol = s, range = r, token = token)
    headers = {
        "Accept" : "application/json",
        "Content-Type" : "application/json"}

    stock_chart_request = requests.get(endpoint, headers = headers)
    stock_chart = stock_chart_request.json()
    
    return stock_chart

# Clean the Data
def getStockChartDatframe(stock, r):
    close = []          # close
    adjClose = []    # fClose
    high = []           # high
    low = []            # lLow
    open = []           # open
    timestamp = []      # date
    volume = []         # volume
    
    chart = getStockChart(IEX_API, stock, r)
    # Extracting only the relevant bits of data from the json object  
    for date in chart:
        for key, value in date.items():    
            if key == "close": close.append(value)
            if key == "fClose": adjClose.append(value)
            if key == "high": high.append(value)
            if key == "low": low.append(value)
            if key == "open": open.append(value)
            if key == "date": timestamp.append(value)
            if key == "volume": volume.append(value)

    stock_chart_dict = {
        "Close" : close,
        "Adj Close" : adjClose,
        "High" : high,
        "Low" : low,
        "Open" : open,
        "Date" : timestamp,
        "Volume" : volume,
    }

    stock_chart_df = pd.DataFrame(data=stock_chart_dict)

    return stock_chart_df

def check_if_valid_chart_data(df: pd.DataFrame) -> bool:
    # Check if dataframe is empty
    if df.empty:
        print("No stock quotes downloaded. Finishing execution")
        return False 

    # Primary Key Check
    if pd.Series(df['Date']).is_unique:
        pass
    else:
        raise Exception("Primary Key check is violated: At least one of the returned stock quotes has the same date")

    # Check for nulls
    if df.isnull().values.any():
        raise Exception("Null values found")

    return True

def get_stock_chart_df(stock):
    # time = '10d'
    time = '1y'
    stock_chart_df = pd.DataFrame(getStockChartDatframe(stock, time), columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])    
    # Validate
    if check_if_valid_chart_data(stock_chart_df):
        print("Data valid, proceed to Load stage")
    else: return pd.DataFrame()

    # Load
    print("Opened database successfully")

    try:
        stock_chart_df.to_sql("{stock}_chart".format(stock = stock), conn, index=False, if_exists='replace')
    except:
        print("Data already exists in the database")
        return pd.DataFrame()
    
    df = pd.read_sql_query("SELECT * FROM {stock}_chart".format(stock = stock), conn, parse_dates=["date"])

    conn.commit()
    print("Close database successfully")
    return df

def getStockQuote(token, symbol):
    endpoint = "https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={token}".format(symbol = symbol, token = token)
    headers = {
        "Accept" : "application/json",
        "Content-Type" : "application/json"}

    stock_quote_request = requests.get(endpoint, headers = headers)
    stock_quote = stock_quote_request.json()
    
    return stock_quote

# Clean the Data
def getStockQuoteData(stocks):
    close = []      # iexClose
    high = []       # week52High
    low = []        # week52Low
    open = []       # iexOpen
    timestamp = []  # latestTime
    volume = []     # iexVolume
    symbols = []    # symbol
    quotes = {}

    for symbol in stocks:
        quotes[symbol] = getStockQuote(IEX_API, symbol)
        # Extracting only the relevant bits of data from the json object  
        for key, value in quotes[symbol].items():    
            if key == "iexClose": close.append(value)
            if key == "week52High": high.append(value)
            if key == "week52Low": low.append(value)
            if key == "iexOpen": open.append(value)
            if key == "latestTime": 
                try: 
                    timestamp.append(datetime.datetime.strptime(value, "%B %d, %Y").strftime("%Y-%m-%d"))
                except ValueError:
                    date = datetime.datetime.now()
                    today = date.strftime("%Y-%m-%d")
                    timestamp.append(today)
            if key == "iexVolume": volume.append(value)
            if key == "symbol": symbols.append(value)

    stock_quote_dict = {
        "Close" : close,
        "High" : high,
        "Low" : low,
        "Open" : open,
        "Date" : timestamp,
        "Volume" : volume,
        "Symbol" : symbols,
    }

    return stock_quote_dict

def check_if_valid_data(df: pd.DataFrame) -> bool:
    # Check if dataframe is empty
    if df.empty:
        print("No stock quotes downloaded. Finishing execution")
        return False 

    # Primary Key Check
    if pd.Series(df['Symbol']).is_unique:
        pass
    else:
        raise Exception("Primary Key check is violated: At least one of the returned stock quotes has the same symblol")

    # Check for nulls
    if df.isnull().values.any():
        print(df.head())
        raise Exception("Null values found")

    return True

def get_stock_df():
    stocks = ['aapl', 'nke']
    stock_quotes = getStockQuoteData(stocks)
    stock_quotes_df = pd.DataFrame(stock_quotes, columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])

    # Validate
    if check_if_valid_data(stock_quotes_df):
        print("Data valid, proceed to Load stage")

    # Load
    print("Opened database successfully")

    try:
        stock_quotes_df.to_sql("my_stock_quotes", conn, index=False, if_exists='replace')
    except:
        print("Data already exists in the database")

    df = pd.read_sql_query('SELECT * FROM my_stock_quotes', conn)

    conn.commit()
    print("Close database successfully")
    return df

if __name__ == '__main__':
    # STOCK_NAME = 'aapl'
    # print(get_stock_chart_df(STOCK_NAME))
    # print(getStockChart(IEX_API, 'aapl', '10d'))
    # print(getStockQuote(IEX_API, 'aapl'))
    # stocks = ['aapl', 'nke']
    # print(getStockQuoteData(stocks))
    print(get_stock_df())
    



