#Flask imports
from flask import Flask, render_template, send_file, make_response, url_for, Response
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

import os
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend, layers
from keras.layers import Dense, Input, RNN

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


CHARTS_FOLDER = os.path.join('static', 'charts')

app = Flask(__name__)
app.config['CHART_FOLDER'] = CHARTS_FOLDER

STOCK_NAME = 'AAPL'
STOCKS = ['AAPL', 'NKE']
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

# STOCK_LIST_DF = pd.DataFrame()
# for stock in STOCKS:
#     STOCK_LIST_DF = get_stock_df('my_stock_list_quotes', stock, TODAY, TODAY)

STOCK_LIST_DF = readSqliteTable('my_stock_list_quotes', None, None)

# print('Here is a list of stock data of the most recent \'weekday\' before today.')
# print(STOCK_LIST_DF.to_numpy())

# # Dataframe to graph out the forcast in dropdown.
# STOCK_DF = get_stock_df('my_stock_quotes', STOCK_NAME, '2014-12-23', TODAY)
STOCK_DF = readSqliteTable('my_stock_quotes', None, None)

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

# @app.route('/results.png')
# def get_results():
if __name__ == '__main__':
    # Extract the input features and target variable
    input_values = STOCK_DF[['Open', 'High', 'Low', 'Close', 'Volume']]
    target_value = STOCK_DF['Close']

    # Calculate the variance of the target variable
    variance = np.var(target_value)
    target_value_summary = target_value.describe()

    X_train, X_test, y_train, y_test = train_test_split(input_values, target_value, test_size=MODEL_TEST_SIZE, random_state=RANDOM_STATE)

    tf.random.set_seed(RANDOM_STATE)

    # Define the model
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l1(REGL1)))
    # model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(REGL2)))
    model.add(tf.keras.layers.Dense(units=64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))
    # Compile the model
    # Create the Adam optimizer with a learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=LOSS)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, shuffle=False)

    # Evaluate the model
    mse = model.evaluate(X_test, y_test, verbose=0)

    print('Here is the summary of the target varable. \n{sum}'.format(sum = target_value_summary))
    print('mse      ', mse)
    print('var      ', variance)
    if mse < (variance*0.66):
        print('Success, the MSE: {mse} is smaller than the Variance: {var}.'.format(mse= mse, var= variance))
    else:
        print('Fail, the MSE: {mse} is not much smaller than the Variance: {var}.'.format(mse= int(mse), var= int(variance)))

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

# sio = SocketIO(app)

# thread = None
# thread_lock = Lock()

# @app.route('/')
# @app.route('/results', methods=("POST", "GET"))
# def index():
#     global thread
#     return render_template('results.html', title='Stock Forcasting', stocks = STOCK_LIST_DF)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80, debug = True)
