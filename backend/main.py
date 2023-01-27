#Flask imports
from flask import Flask, render_template, send_file, make_response, url_for, Response
from flask_socketio import SocketIO

# Data processing imports
import os
import io
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from dateutil.rrule import rrule, DAILY, MO, TU, WE, TH, FR

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras, optimizers
from keras import backend, layers, metrics, Input
from keras.layers import Bidirectional, Dense, Dropout, LSTM, RNN 

# Plotting imports
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Other imports
from datetime import date, datetime, timedelta
from threading import Lock
import math
from scraper import *

app = Flask(__name__)
STOCK_NAME = 'AAPL'
STOCKS = ['AAPL', 'NKE']
YY = '2020'
MM = '01'
DD = '20'

IMAGE_URL = '/backend/static/assets/img/charts/{stock_name}.png'.format(stock_name = STOCK_NAME.lower())

# Get a list of stock data of the most recent 'weekday' before today.
def prev_weekday(adate):
    adate -= datetime.timedelta(days=1)
    while adate.weekday() > 4: # Mon-Fri are 0-4
        adate -= datetime.timedelta(days=1)
    return adate
TODAY = prev_weekday(datetime.datetime.now())
TODAY = TODAY.strftime("%Y-%m-%d")

STOCK_LIST_DF = readSqliteTable('my_stock_list_quotes', None, None)
if STOCK_LIST_DF is None:
    STOCK_LIST_DF = pd.DataFrame()
    for stock in STOCKS:
        STOCK_LIST_DF = get_stock_df('my_stock_list_quotes', stock, TODAY, TODAY)

# print('Here is a list of stock data of the most recent \'weekday\' before today.')
# print(STOCK_LIST_DF.to_numpy())

# # Dataframe to graph out the forcast in dropdown.
STOCK_DF = readSqliteTable('my_stock_quotes', None, None)
if STOCK_DF is None:
    STOCK_DF = pd.DataFrame()
    STOCK_DF = get_stock_df('my_stock_quotes', STOCK_NAME, '{y}-{m}-{d}'.format(y = YY, m = MM, d = DD), TODAY)

# Pre-hyperparameters
DAYS = 30
start_date = datetime.datetime(int(YY), int(MM), int(DD))
end_date = datetime.datetime.strptime(TODAY+' 00:00:00', "%Y-%m-%d %H:%M:%S")
# Generate a range of weekdays (Monday through Friday)
weekdays = list(rrule(DAILY, byweekday=(MO, TU, WE, TH, FR),dtstart=start_date, until=end_date))
weekdays = [(datetime.datetime.strptime(weekday.strftime("%Y-%m-%d"), "%Y-%m-%d")).timestamp() for weekday in weekdays]
future_weekdays = list(rrule(DAILY, byweekday=(MO, TU, WE, TH, FR), count=DAYS, dtstart=end_date))
future_weekdays = [weekday.timestamp() for weekday in future_weekdays]
# print('Last Week Days: ', weekdays[-10:])
# SCALER = StandardScaler()
SCALER = MinMaxScaler()

# Hyperparameters
DROPOUT = 0.18
RANDOM_STATE = 43
MODEL_TEST_SIZE = 23/258
EPOCHS = 30
LEARNING_RATE = 0.0015
REGL1 = 0.01
REGL2 = 0.01
WINDOW_SIZE = 7
WEIGHT = 1.1

ARIMA_P = 2
ARIMA_D = 1
ARIMA_Q = 0

# AutoRegressive Integrated Moving Average Model
class ARIMAModel:
    def __init__(self, y_train, y_val, y_test):
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.model_fit = None

    def fit(self, p, d, q):
        # Fit the ARIMA model to the training data
        self.model = sm.tsa.statespace.SARIMAX(self.y_train, order=(p, d, q))
        self.model_fit = self.model.fit(disp=0)
         
    def forecast(self, days):
        if self.model_fit is None:
            raise ValueError("The model must be fitted before forecasting")
        # Make predictions for the next days
        forecast = self.model_fit.get_forecast(steps=days, alpha=0.05) # with confidence intervals:
        conf_int = forecast.conf_int(alpha=0.05)
        return forecast.predicted_mean, conf_int
        
    def evaluate(self):
        # Make predictions on the validation set
        y_pred_val = self.model_fit.predict(start=len(self.y_train), end=len(self.y_train)+len(self.y_val)-1, dynamic=False)

        # Calculate the mean squared error on the validation set
        mse_val = ((y_pred_val - self.y_val) ** 2).mean()

        # Make predictions on the test set
        y_pred_test = self.model_fit.predict(start=len(self.y_train)+len(self.y_val), end=len(self.y_train)+len(self.y_val)+len(self.y_test)-1, dynamic=False)

        # Calculate the mean squared error on the test set
        mse_test = ((y_pred_test - self.y_test) ** 2).mean()

        return mse_val, mse_test

def preprocess(train, val, test):
    # Normalize the data
    SCALER.fit(train)
    train = SCALER.transform(train)
    val = SCALER.transform(val)
    test = SCALER.transform(test)
    
    return train, val, test

def postprocess(forecasts):
    # Inverse transform the forecasts using the scaler
    forecasts = SCALER.inverse_transform(forecasts)
    return forecasts

# ML Model using Liner Regresion
class StockMLModel(tf.keras.Model):
    def __init__(self):
        super(StockMLModel, self).__init__()
        self.dropout1 = Dropout(DROPOUT)
        self.dense1 = Dense(units=32, activation='relu', input_dim = 5)
        self.dropout2 = Dropout(DROPOUT)
        self.dense2 = Dense(units=16, activation='relu')
        self.dropout3 = Dropout(DROPOUT)
        self.dense3 = Dense(units=1)

    def call(self, inputs):
        x = self.dropout1(inputs)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.dropout3(x)
        return self.dense3(x)

class StockMLModelWithTraining(tf.keras.Model):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super(StockMLModelWithTraining, self).__init__()
        self.model = StockMLModel()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        # self.X_train, self.X_val, self.X_test = preprocess(self.X_train, self.X_val, self.X_test) # Normalize the input features and target variable 
        # self.y_train, self.y_val, self.y_test = preprocess(self.y_train, self.y_val, self.y_test) # Normalize the input features and target variable 

    def compile_and_fit(self):   
        self.X_train, self.X_val, self.X_test = preprocess(self.X_train, self.X_val, self.X_test)
        self.y_train, self.y_val, self.y_test = preprocess(self.y_train, self.y_val, self.y_test)
        # Compile the model
        optimizer = optimizers.RMSprop()

        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])

        # Fit the model to the training data
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=EPOCHS, verbose = 1)

class StockMLModelWithRollingWindow:
    def __init__(self, input_values, target_value, test_size, window_size):
        super(StockMLModelWithRollingWindow, self).__init__()  
        self.model = None
        self.input_values = input_values
        self.target_value = target_value
        self.test_size = test_size
        self.window_size = window_size
    
    def split_data(self):
        # Get the number of rows for the test set
        test_rows = int(self.test_size * self.input_values.shape[0])
        # Get the number of rows for the validation set
        val_rows = int(self.window_size * test_rows)
        # Get the number of rows for the training set
        train_rows = self.input_values.shape[0] - test_rows - val_rows
        # Split the data into training, validation, and test sets
        self.X_train = self.input_values[:train_rows]
        self.y_train = self.target_value[:train_rows]
        self.X_val = self.input_values[train_rows:train_rows+val_rows]
        self.y_val = self.target_value[train_rows:train_rows+val_rows]
        self.X_test = self.input_values[train_rows+val_rows:]
        self.y_test = self.target_value[train_rows+val_rows:]
        self.model = StockMLModelWithTraining(X_train, y_train, X_val, y_val, X_test, y_test)

    def fit_and_evaluate(self):
        self.model.compile_and_fit()

if __name__ == '__main__':
# @app.route('/forecast.png')
# def get_forecast():
    # Extract the input features and target variable with GCD of rows to WINDOW_SIZE, hence '.iloc[5:]'.
    input_values = STOCK_DF[['Date', 'Open', 'High', 'Low', 'Volume']]
    # target_value = STOCK_DF[['Close', 'Low']]
    target_value = STOCK_DF[['Close']]

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(input_values, target_value, test_size=MODEL_TEST_SIZE, random_state=RANDOM_STATE)   # 90-iv, 10-iv, 90-tv, 10-tv
    # X_train.iloc[:, -1] = X_train.iloc[:, -1] * WEIGHT
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=MODEL_TEST_SIZE, random_state=RANDOM_STATE)               # 81-iv, 9-iv, 81-tv, 9-tv
    
    # Output the input_values and target_value info and calculations
    variance = np.var(target_value)
    # variance = np.var(target_value['Close'])
    print('The model input values for {stock} are as follows:\n{head}\n'.format(stock = STOCK_NAME, head = input_values.head()))
    print('The model target values for {stock} are as follows:\n{head}\nvariance: {var}\n{sum}\n'.format(stock = STOCK_NAME, head = target_value.head(), var = variance, sum = target_value.describe()))

    # print('\nX_train:\n',X_train)
    # print('\nX_val:\n',X_val)
    # print('\ny_train:\n',y_train)
    # print('\ny_val:\n',y_val)
    # print('\nX_test:\n',X_test)
    # print('\ny_test:\n',y_test)
        
    # Initialize the machine learning model, then compile 
    stock_model = StockMLModelWithTraining(X_train, y_train, X_val, y_val, X_test, y_test)
    stock_model.compile_and_fit()
    ml_model = stock_model.model
    ml_forecasting = ml_model.predict(X_val)
    ml_forecasting = postprocess(ml_forecasting)
    # print('The ML Model Forecast:\n', ml_forecasting)

    # Initialize the StockMLModelWithRollingWindow class with the data and test_size
    # stock_model = StockMLModelWithRollingWindow(input_values, target_value,TEST_SIZE, WINDOW_SIZE)
    # stock_model.split_data()
    # stock_model.fit_and_evaluate()
    # ml_model = stock_model.model.model
    # ml_forecasting = ml_model.predict(stock_model.X_val)
    # ml_forecasting = postprocess(ml_forecasting)

    # Instantiate the model with the training, validation, and test sets
    arima_model = ARIMAModel(y_train, y_val, y_test)
    arima_model.fit(ARIMA_P, ARIMA_D, ARIMA_Q)
    arima__forecasting, conf_int = arima_model.forecast(days=DAYS)
    print("The ARIMA Model Forecast:\n", arima__forecasting)

    # Evaluate the model
    mse = mean_squared_error(y_val, ml_forecasting)
    result = ''
    if mse < (variance*0.66):
        result += 'Success, the MSE: {mse} is smaller than the Variance: {var}.'.format(mse= int(mse), var= int(variance))
    else:
        result += 'Fail, the MSE: {mse} is not much smaller than the Variance: {var}.'.format(mse= int(mse), var= int(variance))
    print(result)

    # Fit the model with p, d, and q values
    # arima_model.fit(ARIMA_P, ARIMA_D, ARIMA_Q)
    # mse_val, mse_test = arima_model.evaluate()

    # Plot the forecast and the true values
#     fig, ax = plt.subplots(figsize = (25,8))
#     cd = target_value[-20:]
#     ax.plot(weekdays[-len(cd):], cd, label='Current Data', color='black')
#     # ax.plot(weekdays[-len(cd):], y_val[-20:], label='Y VAL Vales', color='green')

#     ax.plot(future_weekdays, ml_forecasting[:DAYS], label='ML Forecast', color='blue')
#     ax.plot([], [], label=result, alpha=0)
#     ax.plot([], [], label='', alpha=0)
#     # plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink')

#     # ax.plot(future_weekdays, arima__forecasting, label='AMIRA Forecast')
#     # # ax.plot([], [], label='', alpha=0)
#     # ax.plot([], [], label='ARIMA Model: MSE with the test set is {t} and with the validation set is {v}'.format(t = int(mse_test), v= int(mse_val)), alpha=0)

   
#     ax.set_ylabel('Stock Price (USD)')
#     ax.set_xlabel('Date (YYYY-MM-DD)')
    
#     # Add coordinates to the plot
#     for x, y in zip(weekdays[-len(cd):], cd):
#         if x % 5 == 0:
#             ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='black', weight='bold')

#     # for x, y in zip(future_weekdays, arima__forecasting):
#     #     ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red', weight='bold')

#     for x, y in zip(future_weekdays, ml_forecasting[:DAYS]):
#         ax.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='blue', weight='bold')


#     ax.legend()
#     ax.set_title('Stock Price Forecast')
    
#     # Save the plot to a temporary file
#     fig.savefig('temp.png')
    
#     # Send the plot back to the client
#     return send_file('temp.png', mimetype='image/png')

# sio = SocketIO(app)

# thread = None
# thread_lock = Lock()

# @app.route('/')
# @app.route('/results', methods=("POST", "GET"))
# def index():
#     global thread
#     return render_template('results.html', title='Stock Forcasting', stocks = STOCK_LIST_DF, today = TODAY)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80, debug = True)
# 79468220312269963264
# 8325919859938426880
# 5887356508577938276352 -> 7
# 33009899080748703416320 -> 8
# 12905441533398978920448 -> 14