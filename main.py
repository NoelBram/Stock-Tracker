from contextlib import nullcontext
import copy
from distutils.log import error
from pyexpat.model import XML_CTYPE_ANY
import re
import sqlite3
from textwrap import indent
from xml.etree.ElementTree import tostring
# from flask import Flask
import sqlalchemy
import pandas as pd 
from sqlalchemy.orm import sessionmaker
import requests
import json
from datetime import datetime
import datetime
# from keyvalue_sqlite import KeyValueSqlite
import base64

IEX_API = "pk_913ba7d52f144907a92856b52ea0636e"
DATABASE_LOCATION = "sqlite:///my_stock_quotes.sqlite"


def getStockQuote(token, symbol):
    
    endpoint = "https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={token}".format(symbol = symbol, token = token)
    headers = {
        "Accept" : "application/json",
        "Content-Type" : "application/json"}

    stock_quote_request = requests.get(endpoint, headers = headers)
    stock_quote = stock_quote_request.json()
    
    return stock_quote

def getStockQuoteData(stocks):
    date = datetime.datetime.now()
    date_y_m_d = date.strftime("%Y-%m-%d")

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
            if key == "latestTime": timestamp.append(date_y_m_d)
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
        raise Exception("Primary Key check is violated")

    # Check for nulls
    if df.isnull().values.any():
        raise Exception("Null values found")

    # Check that all timestamps are of yesterday's date
    today = datetime.datetime.now()
    today = today.strftime("%Y-%m-%d")


    timestamps = df["Date"].tolist()
    for timestamp in timestamps:
        if datetime.datetime.strptime(timestamp, '%Y-%m-%d') == today:
            raise Exception("At least one of the returned stock quotes does not have a today's timestamp")

    return True


if __name__ == "__main__":
    stocks = ['aapl', 'nke']
    stock_quotes = getStockQuoteData(stocks)
    stock_quotes_df = pd.DataFrame(stock_quotes, columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    # Validate
    if check_if_valid_data(stock_quotes_df):
        print("Data valid, proceed to Load stage")

    # Load
    engine = sqlalchemy.create_engine(DATABASE_LOCATION)
    conn = sqlite3.connect('my_stock_quotes.sqlite')
    cursor = conn.cursor()

    print("Opened database successfully")

    try:
        stock_quotes_df.to_sql("my_stock_quotes", conn, index=False, if_exists='replace')
    except:
        print("Data already exists in the database")

    # df = pd.read_sql_query('SELECT * FROM my_stock_quotes', conn, parse_dates=["symbol"])

    conn.close()
    print("Close database successfully")
    # print(df.head)

