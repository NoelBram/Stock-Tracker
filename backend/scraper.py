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
import sqlite3
from textwrap import indent
from xml.etree.ElementTree import tostring

IEX_API = "pk_913ba7d52f144907a92856b52ea0636e"
DATABASE_LOCATION = "sqlite:///db.sqlite"

def getStockChart(token, s, r):
    endpoint = "https://cloud.iexapis.com/stable/stock/{symbol}/chart/{range}?token={token}&chartByDay=true".format(symbol = s, range = r, token = token)
    headers = {
        "Accept" : "application/json",
        "Content-Type" : "application/json"}

    stock_chart_request = requests.get(endpoint, headers = headers)
    stock_chart = stock_chart_request.json()
    
    return stock_chart

def getStockChartData(stock, r):
    close = []      # iexClose
    high = []       # week52High
    low = []        # week52Low
    open = []       # iexOpen
    timestamp = []  # latestTime
    volume = []     # iexVolume
    symbols = []    # symbol
    
    chart = getStockChart(IEX_API, stock, r)
    # Extracting only the relevant bits of data from the json object  
    for date in chart:
        for key, value in date.items():    
            if key == "close": close.append(value)
            if key == "high": high.append(value)
            if key == "low": low.append(value)
            if key == "open": open.append(value)
            if key == "date": timestamp.append(value)
            if key == "volume": volume.append(value)
            if key == "symbol": symbols.append(value)

    stock_chart_dict = {
        "Close" : close,
        "High" : high,
        "Low" : low,
        "Open" : open,
        "Date" : timestamp,
        "Volume" : volume,
        "Symbol" : symbols,
    }

    return stock_chart_dict

def check_if_valid_chart_data(df: pd.DataFrame) -> bool:
    # Check if dataframe is empty
    if df.empty:
        print("No stock quotes downloaded. Finishing execution")
        return False 

    # Primary Key Check
    if pd.Series(df['Date']).is_unique:
        pass
    else:
        raise Exception("Primary Key check is violated")

    # Check for nulls
    if df.isnull().values.any():
        raise Exception("Null values found")

    # Check that all timestamps are of yesterday's date
    today = datetime.datetime.now()
    today = today.strftime("%Y-%m-%d")


    timestamps = df["Date"].tolist()
    for timestamp in timestamps:
        if timestamp != today:
            raise Exception("At least one of the returned stock quotes does not have a today's timestamp")

    return True


def load_chart_dataframe():
    stocks = ['aapl', 'nke']
    week_range = '5d'
    stock_quotes = {}
    for stock in stocks:
        stock_quotes[stock] = getStockChartData(stock, week_range)
    
    df = pd.concat([pd.DataFrame([stock_quotes[stocks[i]]], columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]) for i in range(0, len(stocks))],ignore_index=True)


    # # Validate
    # if check_if_valid_chart_data(stock_charts_df):
    #     print("Data valid, proceed to Load stage")

    # # Load
    # conn = sqlite3.connect('db.sqlite')

    # print("Opened database successfully")

    # try:
    #     stock_charts_df.to_sql("my_stock_charts", conn, index=False, if_exists='replace')
    # except:
    #     print("Data already exists in the database")

    # df = pd.read_sql_query('SELECT * FROM my_stock_charts', conn, parse_dates=["date"])

    # conn.close()
    # print("Close database successfully")
    return df

# def getStockQuote(token, symbol):
#     endpoint = "https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={token}".format(symbol = symbol, token = token)
#     headers = {
#         "Accept" : "application/json",
#         "Content-Type" : "application/json"}

#     stock_quote_request = requests.get(endpoint, headers = headers)
#     stock_quote = stock_quote_request.json()
    
#     return stock_quote

# def getStockQuoteData(stocks):
#     close = []      # iexClose
#     high = []       # week52High
#     low = []        # week52Low
#     open = []       # iexOpen
#     timestamp = []  # latestTime
#     volume = []     # iexVolume
#     symbols = []    # symbol
#     quotes = {}

#     for symbol in stocks:
#         quotes[symbol] = getStockQuote(IEX_API, symbol)
#         # Extracting only the relevant bits of data from the json object  
#         for key, value in quotes[symbol].items():    
#             if key == "iexClose": close.append(value)
#             if key == "week52High": high.append(value)
#             if key == "week52Low": low.append(value)
#             if key == "iexOpen": open.append(value)
#             if key == "latestTime": timestamp.append(datetime.datetime.strptime(value, "%B %d, %Y").strftime("%Y-%m-%d"))
#             if key == "iexVolume": volume.append(value)
#             if key == "symbol": symbols.append(value)

#     stock_quote_dict = {
#         "Close" : close,
#         "High" : high,
#         "Low" : low,
#         "Open" : open,
#         "Date" : timestamp,
#         "Volume" : volume,
#         "Symbol" : symbols,
#     }

#     return stock_quote_dict

# def check_if_valid_data(df: pd.DataFrame) -> bool:
#     # Check if dataframe is empty
#     if df.empty:
#         print("No stock quotes downloaded. Finishing execution")
#         return False 

#     # Primary Key Check
#     if pd.Series(df['Symbol']).is_unique:
#         pass
#     else:
#         raise Exception("Primary Key check is violated")

#     # Check for nulls
#     if df.isnull().values.any():
#         raise Exception("Null values found")

#     # Check that all timestamps are of yesterday's date
#     today = datetime.datetime.now()
#     today = today.strftime("%Y-%m-%d")


#     timestamps = df["Date"].tolist()
#     for timestamp in timestamps:
#         if timestamp != today:
#             raise Exception("At least one of the returned stock quotes does not have a today's timestamp")

#     return True

# def load_dataframe():
#     stocks = ['aapl', 'nke']
#     week_range = '5dm'
#     stock_quotes = getStockQuoteData(stocks)
#     stock_quotes_df = pd.DataFrame(stock_quotes, columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
#     # Validate
#     if check_if_valid_data(stock_quotes_df):
#         print("Data valid, proceed to Load stage")

#     # Load
#     conn = sqlite3.connect('db.sqlite')

#     print("Opened database successfully")

#     try:
#         stock_quotes_df.to_sql("my_stock_quotes", conn, index=False, if_exists='replace')
#     except:
#         print("Data already exists in the database")

#     df = pd.read_sql_query('SELECT * FROM my_stock_quotes', conn, parse_dates=["symbol"])

#     conn.close()
#     print("Close database successfully")
#     return df

if __name__ == '__main__':
    # print(load_dataframe().to_numpy())
    # print(load_chart_dataframe().to_numpy())
    print(load_chart_dataframe().head())

