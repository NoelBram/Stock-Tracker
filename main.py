from contextlib import nullcontext
import copy
from distutils.log import error
from pyexpat.model import XML_CTYPE_ANY
import re
import sqlite3
from textwrap import indent
from xml.etree.ElementTree import tostring
# from flask import Flask
# import sqlalchemy
import pandas as pd 
# from sqlalchemy.orm import sessionmaker
import requests
import json
from datetime import datetime
import datetime
# from keyvalue_sqlite import KeyValueSqlite
import base64

IEX_API = "pk_913ba7d52f144907a92856b52ea0636e"

def getStockQuote(token, symbol):
    
    endpoint = "https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={token}".format(symbol = symbol, token = token)
    headers = {
        "Accept" : "application/json",
        "Content-Type" : "application/json"}

    stock_quote_request = requests.get(endpoint, headers = headers)
    stock_quote = stock_quote_request.json()
    
    return stock_quote

def getStockQuoteData(stocks):
    timestamps = datetime.datetime.now()
    timestamps_y_m_d = timestamps.strftime("%Y-%m-%d")

    close = []  # iexClose
    high = []   # week52High
    low = []    # week52Low
    open = []   # iexOpen
    time = []   # latestTime
    volume = [] # iexVolume
    symbols = [] # symbol
    
    quotes = {}
    for symbol in stocks:
        quotes[symbol] = getStockQuote(IEX_API, symbol)
        # Extracting only the relevant bits of data from the json object  
        for key, value in quotes[symbol].items():    
            if key == "iexClose": close.append(value)
            if key == "week52High": high.append(value)
            if key == "week52Low": low.append(value)
            if key == "iexOpen": open.append(value)
            if key == "latestTime": time.append(timestamps_y_m_d)
            if key == "iexVolume": volume.append(value)
            if key == "symbol": symbols.append(value)

    
    
    
    stock_quote_dict = {
        "close" : close,
        "high" : high,
        "low" : low,
        "open" : open,
        "time" : time,
        "volume" : volume,
    }

    return stock_quote_dict

if __name__ == "__main__":
    # print(getStockQuote(IEX_API, "aapl"))
    stocks = ['aapl', 'nke']
    print(getStockQuoteData(stocks))

