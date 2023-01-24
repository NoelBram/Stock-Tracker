#Flask imports
from flask import Flask, render_template, send_file, make_response, url_for, Response
from flask_socketio import SocketIO

# Data processing imports
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from dateutil.rrule import rrule, DAILY, MO, TU, WE, TH, FR

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from keras import backend, layers, metrics
from keras.layers import Bidirectional, Dense, Dropout, Input, Bidirectional, RNN 
from keras.losses import mean_squared_error

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
YY = '2021'
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

# Hyperparameters
TEST_SIZE = 60
RANDOM_STATE = 42
LOSS = 'mean_squared_error'
MODEL_TEST_SIZE = 30/258
EPOCHS = 50
LEARNING_RATE = 0.000055
REGL1 = 0.01
REGL2 = 0.01
WINDOW_SIZE = 30
WEIGHT = 1.000001

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

# ML Model using Liner Regresion
class StockMLModel(tf.keras.Model):
    def __init__(self):
        super(StockMLModel, self).__init__()
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=64, activation='relu')
        self.dense4 = Dense(units=1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

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

    def compile_and_fit(self, optimizer, loss, epochs):
        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mse'])

        # Fit the model to the training data
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs, shuffle=False)

def forecast(self, future_forecast_days):
    # Get the last day of the training data
    last_day_of_train_data = self.X_train['Date'].iloc[-1]
    # Create a dataframe for the future forecast
    future_forecast = pd.DataFrame()
    # Add the dates for the future forecast
    future_forecast['Date'] = pd.date_range(last_day_of_train_data, periods=future_forecast_days+1, freq='D')[1:]
    # Add the symbols for the future forecast
    future_forecast['Symbol'] = self.X_train['Symbol'].iloc[0]
    # Merge the input data for the future forecast with the X_train data
    future_forecast = pd.merge(future_forecast, self.X_train, on='Symbol')
    # Drop the 'Date' and 'Symbol' columns from the input data
    future_forecast.drop(columns=['Date', 'Symbol'], inplace=True)
    # Use the model to predict the closing price for the future forecast days
    future_forecast['Close'] = self.model.predict(future_forecast)
    return future_forecast

def preprocess(X, y):
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Impute missing values
    X = pd.DataFrame(X).fillna(0).values
    
    return X, y

def postprocess(forecasts):
    # Inverse transform the forecasts using the scaler
    scaler = StandardScaler()
    forecasts = scaler.inverse_transform(forecasts)
    
    return forecasts

# if __name__ == '__main__':
@app.route('/forecast.png')
def get_forecast():
    # Extract the input features and target variable with GCD of rows to WINDOW_SIZE, hence '.iloc[5:]'.
    input_values = STOCK_DF[['Open', 'High', 'Low', 'Close', 'Volume', 'Date']].iloc[5:]
    target_value = STOCK_DF['Close'].iloc[5:]

    # Calculate the variance of the target variable
    variance = np.var(target_value)
    print('The model input values for {stock} are as follows:\n{sum}\n'.format(stock = STOCK_NAME, sum = input_values.describe()))
    print('The model target values for {stock} are as follows:\nvariance: {var}\n{sum}\n'.format(stock = STOCK_NAME, var = variance, sum = target_value.describe()))

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(input_values, target_value, test_size=0.1, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
    # Add more weight to the last input data

    tf.random.set_seed(RANDOM_STATE)
    
    # Create the Adam optimizer with a learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Initialize the machine learning model, then compile 
    stock_model = StockMLModelWithTraining(X_train, y_train, X_val, y_val, X_test, y_test)
    stock_model.compile_and_fit(optimizer, LOSS, EPOCHS)
    ml_model = stock_model.model
    ml_forecasting = ml_model.predict(X_val)
    # print('The ML Model Forecast:\n', ml_forecasting)

    # Instantiate the model with the training, validation, and test sets
    arima_model = ARIMAModel(y_train, y_val, y_test)
    arima_model.fit(ARIMA_P, ARIMA_D, ARIMA_Q)
    arima__forecasting, conf_int = arima_model.forecast(days=DAYS)
    # print("The ARIMA Model Forecast:\n", arima__forecasting)

    # Evaluate the model
    mse = np.mean(ml_model.evaluate(X_val, y_val, verbose=0))
    print('The Scores are ', mse)
    result = ''
    if mse < (variance*0.66):
        result += 'Success, the MSE: {mse} is smaller than the Variance: {var}.'.format(mse= int(mse), var= int(variance))
    else:
        result += 'Fail, the MSE: {mse} is not much smaller than the Variance: {var}.'.format(mse= int(mse), var= int(variance))

    # Fit the model with p, d, and q values
    arima_model.fit(ARIMA_P, ARIMA_D, ARIMA_Q)
    mse_val, mse_test = arima_model.evaluate()

    # Plot the forecast and the true values
    fig, ax = plt.subplots(figsize = (25,8))
    cd = target_value[-20:]
    ax.plot(weekdays[-len(cd):], cd, label='Current Data', color='black')
    # ax.plot(weekdays[-len(cd):], y_val[-20:], label='Y VAL Vales', color='green')

    ax.plot(future_weekdays, ml_forecasting[:DAYS], label='ML Forecast', color='blue')
    ax.plot([], [], label=result, alpha=0)
    ax.plot([], [], label='', alpha=0)
    # plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink')

    ax.plot(future_weekdays, arima__forecasting, label='AMIRA Forecast')
    # ax.plot([], [], label='', alpha=0)
    ax.plot([], [], label='ARIMA Model: MSE with the test set is {t} and with the validation set is {v}'.format(t = int(mse_test), v= int(mse_val)), alpha=0)

   
    ax.set_ylabel('Stock Price (USD)')
    ax.set_xlabel('Date (YYYY-MM-DD)')
    
    ax.legend()

    ax.set_title('Stock Price Forecast')
    
    # Save the plot to a temporary file
    fig.savefig('temp.png')
    
    # Send the plot back to the client
    return send_file('temp.png', mimetype='image/png')

sio = SocketIO(app)

thread = None
thread_lock = Lock()

@app.route('/')
@app.route('/results', methods=("POST", "GET"))
def index():
    global thread
    return render_template('results.html', title='Stock Forcasting', stocks = STOCK_LIST_DF, today = TODAY)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug = True)
