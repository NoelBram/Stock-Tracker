#Flask imports
from flask import Flask, render_template, send_file, make_response, url_for, Response
import hyperopt
from hyperopt import fmin, tpe, hp

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend, layers, metrics
from keras.layers import Bidirectional, Dense, Dropout, Input, Bidirectional, RNN 
from keras.losses import mean_squared_error

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import date, datetime, timedelta
from threading import Lock
from flask import Flask, render_template
from flask_socketio import SocketIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from flask_sqlalchemy import SQLAlchemy
from scraper import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from dateutil.rrule import rrule, DAILY, MO, TU, WE, TH, FR

CHARTS_FOLDER = os.path.join('static', 'charts')

app = Flask(__name__)
app.config['CHART_FOLDER'] = CHARTS_FOLDER

STOCK_NAME = 'AAPL'
STOCKS = ['AAPL', 'NKE']
YY = '2022'
MM = '11'
DD = '02'

IMAGE_URL = '/backend/static/assets/img/charts/{stock_name}.png'.format(stock_name = STOCK_NAME.lower())

# Get a list of stock data of the most recent 'weekday' before today.
def prev_weekday(adate):
    adate -= datetime.timedelta(days=1)
    while adate.weekday() > 4: # Mon-Fri are 0-4
        adate -= datetime.timedelta(days=1)
    return adate
TODAY = prev_weekday(datetime.datetime.now())
TODAY = TODAY.strftime("%Y-%m-%d")
print('\nHere is most recent weekday before today.: {day}\n'.format (day = TODAY))

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

# Set the start date and end date
start_date = datetime.datetime(int(YY), int(MM), int(DD))
end_date = datetime.datetime(2023, 1, 6)

# Pre-hyperparameters
DAYS = 30

# Generate a range of weekdays (Monday through Friday)
weekdays = list(rrule(DAILY, byweekday=(MO, TU, WE, TH, FR),dtstart=start_date, until=end_date))
weekdays = [(datetime.datetime.strptime(weekday.strftime("%Y-%m-%d"), "%Y-%m-%d")).timestamp() for weekday in weekdays]
future_weekdays = list(rrule(DAILY, byweekday=(MO, TU, WE, TH, FR), count=DAYS, dtstart=end_date))
future_weekdays = [weekday.timestamp() for weekday in future_weekdays]

# print('Last Week Days: ', weekdays[-10:])

print('Here is stock data for {stock}.'.format(stock = STOCK_NAME))
print(STOCK_DF.to_numpy())
print(STOCK_DF.info())

# Hyperparameters
TEST_SIZE = 30
RANDOM_STATE = 42
LOSS = 'mean_squared_error'
MODEL_TEST_SIZE = TEST_SIZE/258
EPOCHS = 50
LEARNING_RATE = 0.000055
REGL1 = 0.01
REGL2 = 0.01
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
        forecast = self.model_fit.get_forecast(days=days, alpha=0.05) # with confidence intervals:
        conf_int = forecast.conf_int(alpha=0.05)
        return forecast.prediction_mean, conf_int
        
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
    # Extract the input features and target variable
    input_values = STOCK_DF[['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
    target_value = STOCK_DF['Close']

    # Calculate the variance of the target variable
    variance = np.var(target_value)
    target_value_summary = target_value.describe()
    
    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(input_values, target_value, test_size=0.1, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)
    # Add more weight to the last input data
    X_train.iloc[:, -1] = X_train.iloc[:, -1] * WEIGHT

    tf.random.set_seed(RANDOM_STATE)
    
    # Create the Adam optimizer with a learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Initialize the machine learning model, then compile 
    stock_model = StockMLModelWithTraining(X_train, y_train, X_val, y_val, X_test, y_test)
    stock_model.compile_and_fit(optimizer, LOSS, EPOCHS)

    ml_model = stock_model.model
    ml_forecasting = ml_model.predict(X_val)

    print('Next Week Forecasting: \n', ml_forecasting)

    # Instantiate the model with the training, validation, and test sets
    arima_model = ARIMAModel(y_train, y_val, y_test)

    arima_model.fit(ARIMA_P, ARIMA_D, ARIMA_Q)
    arima__forecasting, conf_int = arima_model.forecast(days=DAYS)
    print("The arima_forecast: ", arima__forecasting)

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
    # ax.plot(weekdays[-len(cd):], cd, label='Current Data')

    # ax.plot([], [], label='ML Model: Mean Squared Error:  %.4f'%mse, alpha=0)
    # ax.plot(future_weekdays, ml_forecasting[:DAYS], label='Next Month\'s Forecast')
    # ax.plot([], [], label='', alpha=0)

    ax.plot([], [], label='ARIMA Model: Mean Squared Error on test set is {t} validation set is {v}'.format(t = int(mse_test), v= int(mse_val)), alpha=0)
    plt.plot(weekdays[-len(cd):], y_test[-20:], label='Actual', color='red')
    plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink')
    plt.plot(future_weekdays, arima__forecasting, label='forecast')
    ax.plot([], [], label='', alpha=0)

    ax.plot([], [], label=result, alpha=0)
   
    ax.set_ylabel('Stock Price (USD)')
    ax.set_xlabel('Date (YYYY-MM-DD)')
    
    ax.legend()

    ax.set_title('Stock Price Forecast')
    
    # Save the plot to a temporary file
    fig.savefig('temp.png')
    
    # Send the plot back to the client
    return send_file('temp.png', mimetype='image/png')

    # Evaluate the model
    # mse = ml_model.evaluate(X_test, y_test, verbose=0)

    # print('Here is the summary of the target varable. \n{sum}'.format(sum = target_value_summary))
    # print('mse      ', mse)
    # print('var      ', variance)

# if __name__ == '__main__':
#     # Define the search space for the hyperparameters
#     space = hp.choice('model_type', [
#         {
#             'type': 'linear',
#             'regularization': hp.lognormal('regularization', 0, 1)
#         },
#         {
#             'type': 'random_forest',
#             'max_depth': hp.quniform('max_depth', 2, 10, 1),
#             'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
#         }
#     ])

#     # Define the objective function to be minimized
#     def objective(params):
#         model = create_model(params)  # Function to create the model with the specified hyperparameters
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_val)
#         return mean_squared_error(y_val, y_pred)

#     # Run the optimization
#     best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

#     # Use the optimal hyperparameters to create the final model
#     optimal_model = create_model(best)
#     optimal_model.fit(X_train, y_train)
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

#     fig = create_figure(STOCK_DF)

#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
    
#     return Response(output.getvalue(), mimetype='image/png')

# def create_figure(df): 
#     fig, ax = plt.subplots(figsize = (6,4))
#     fig.patch.set_facecolor('#E8E5DA')

#     loss = df.history['loss']
#     val_loss = df.history['val_loss']

#     ax.bar(loss, val_loss, color = "#304C89")

#     plt.ylabel('Stock Price (USD)', fontsize=8)
#     plt.xlabel('Date (YYYY-MM-DD)', fontsize=8)
#     plt.plot(EPOCHS, loss, 'r')
#     plt.plot(EPOCHS, val_loss, 'b')
#     plt.title('Training and validation accuracy')
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend(["Accuracy", "Validation Accuracy"])

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
