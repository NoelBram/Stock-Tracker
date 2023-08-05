#Flask imports
from flask import Flask, render_template

# Data processing imports
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import numpy as np, array
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
from keras.models import Sequential

# Plotting imports
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Other imports
from datetime import date, datetime, timedelta
from threading import Lock
from scraper import *
import seaborn as sns
import matplotlib.pyplot as plt


app = Flask(__name__)
STOCKS = ['AAPL', 'NKE']
YY = '2022'
MM = '06'
DD = '16'

# Get a list of stock data of the most recent 'weekday' before today.
def prev_weekday(adate):
    adate -= datetime.timedelta(days=1)
    while adate.weekday() > 4: # Mon-Fri are 0-4
        adate -= datetime.timedelta(days=1)
    return adate
TODAY = prev_weekday(datetime.datetime.utcnow())
TODAY = TODAY.strftime("%Y-%m-%d")

STOCK_LIST_DF = pd.DataFrame()
    
# print('Here is a list of stock data of the most recent \'weekday\' before today.')
# print(STOCK_LIST_DF.to_numpy())

# Pre-hyperparameters
DAYS = 30
start_date = datetime.datetime(int(YY), int(MM), int(DD))
end_date = datetime.datetime.strptime(TODAY+' 00:00:00', "%Y-%m-%d %H:%M:%S")
# Generate a range of weekdays (Monday through Friday)
WEEKDAYS = list(rrule(DAILY, byweekday=(MO, TU, WE, TH, FR),dtstart=start_date, until=end_date))
WEEKDAYS = [(datetime.datetime.strptime(weekday.strftime("%Y-%m-%d"), "%Y-%m-%d")).timestamp() for weekday in WEEKDAYS]
future_weekdays = list(rrule(DAILY, byweekday=(MO, TU, WE, TH, FR), count=DAYS, dtstart=end_date))
future_weekdays = [weekday.timestamp() for weekday in future_weekdays]
# print('Last Week Days: ', weekdays[-10:])
# SCALER = StandardScaler()
SCALER = MinMaxScaler()
Y_VALUE = pd.DataFrame()

# Hyperparameters
DROPOUT = 0.18
RANDOM_STATE = 43
TEST_SIZE = 0.80
MODEL_TEST_SIZE = 23/258
EPOCHS = 30
LEARNING_RATE = 0.0015
REGL1 = 0.01
REGL2 = 0.01
WINDOW_SIZE = 0
N_FEATURES = 0
BATCH_SIZE = 14

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

# ML Model using Liner Regression
class StockMLModel(tf.keras.Model):
    def __init__(self):
        super(StockMLModel, self).__init__()
        self.dropout1 = Dropout(DROPOUT)
        self.lstm = LSTM(units=64,return_sequences=True, return_state=True, activation='relu')
        self.dense1 = Dense(units=32, activation='relu')
        self.dropout2 = Dropout(DROPOUT)
        self.dense2 = Dense(units=16, activation='relu')
        self.dropout3 = Dropout(DROPOUT)
        self.dense3 = Dense(units=1)

    def call(self, inputs):
        x = self.dropout1(inputs)
        # x = self.lstm(x)
        x = self.dense1(x[:, -1, :]) # extract last output of LSTM sequence
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.dropout3(x)
        return self.dense3(x)

class StockMLModelWithTraining(tf.keras.Model):
    def __init__(self, input_values, y, train_ratio, val_ratio):
        super(StockMLModelWithTraining, self).__init__()
        self.input_values = input_values
        self.y = y
        self.tr = train_ratio
        self.vr = val_ratio
        self.model = StockMLModel()
        
    def compile_and_fit(self):
        # Get the number of rows for each set
        train_rows = int(train_pct * len(self.input_values))
        val_rows = int(val_pct * len(self.input_values))

        # Split the data into training, validation, and test sets
        X_train = self.input_values[:train_rows]
        y_train = self.y[:train_rows]
        X_val = self.input_values[train_rows:train_rows+val_rows]
        y_val = self.y[train_rows:train_rows+val_rows]
        X_test = self.input_values[train_rows+val_rows:]
        y_test = self.y[train_rows+val_rows:]

        # Verify the shapes of the datasets
        print("Training set shape: ", X_train.shape, y_train.shape)
        print("Validation set shape: ", X_val.shape, y_val.shape)
        print("Test set shape: ", X_test.shape, y_test.shape)
        # print(y_train)

        # # Prepare input sequences
        # WINDOW_SIZE = 6
        # X = []
        # for i in range(WINDOW_SIZE, len(self.input_values)):
        #     X.append(self.input_values[i-WINDOW_SIZE:i])
        # self.y = self.y[WINDOW_SIZE:]  # only keep target values starting from the nth step
        # N_FEATURES = np.array(X)

        # n = N_FEATURES.shape[0]
        # train_size = int(n * self.tr)
        # val_size = int(n * self.vr)

        # X_train = self.input_values[:train_size]
        # y_train = self.y[:train_size]
        # X_val = self.input_values[train_size:train_size + val_size]
        # y_val = self.y[train_size:train_size + val_size]
        # X_test = self.input_values[train_size + val_size:]
        # y_test = self.y[train_size + val_size:]

        # # return X_train, y_train, X_val, y_val, X_test, y_test

        # X_train, X_val, X_test = preprocess(X_train, X_val, X_test) # Normalize the input features and target variable 
        # y_train, y_val, y_test = preprocess(y_train.reshape(-1, 1), y_val.reshape(-1, 1), y_test.reshape(-1, 1)) # Normalize the target variable 
       
        # # Compile the model
        # optimizer = optimizers.RMSprop()

        # self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])

        # # Fit the model to the training data
        # self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose = 0)
        # # self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, verbose = 0)

        
    # def compile_and_fit_test(self):
    #     # Compile the model
    #     optimizer = optimizers.RMSprop()
    #     self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])

    #     # Fit the model to the training data
    #     self.model.fit(self.X_train, self.y_train, epochs=EPOCHS, verbose=0)

    #     # Evaluate the model on the test data
    #     test_loss, test_acc, test_mse = self.model.evaluate(self.X_test, self.y_test, verbose=0)

    #     print('Test loss:', test_loss)
    #     print('Test accuracy:', test_acc)
    #     print('Test MSE:', test_mse)

def save_model(model):
    # Get the current time
    now = datetime.datetime.utcnow()
    now = now.strftime("%Y-%m-%d_%H:%M:%S")
    folder = f'ml_model_{now}'
    tf.saved_model.save(model, folder)
    return folder

def split_sequences(df, n_steps):
    X, y = [], []
    for i in range(len(df)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(df)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = df.iloc[i:end_ix, :-1].values, df.iloc[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# @app.route('/forecast.png')
# def get_forecast():
#     # Extract the input features and target variable with GCD of rows to WINDOW_SIZE, hence '.iloc[5:]'.

#     # Split input data into X and y
#     input_values = STOCK_DF[['Date', 'Open', 'High', 'Low', 'Volume']].values
#     Y_VALUE = STOCK_DF['Close'].values  # replace 'Target_Column' with the name of the target column

#     # Define the percentages for each set
#     train_pct = 0.7
#     val_pct = 0.2
 
#     # Define model
#     model = StockMLModelWithTraining(input_values, Y_VALUE, train_pct, val_pct)
#     model.compile_and_fit()
#     model = model.model

    # print(model.summary)

    # N_FEATURES = X.shape[2]
    # # Split the data into train, validation, and test sets
    # train_size = 0.7
    # val_size = 0.1

    # num_rows = data.shape[0]
    # train_end = int(train_size * num_rows)
    # val_end = train_end + int(val_size * num_rows)

    # # Train set
    # X_train = data.iloc[:train_end, :-1].values
    # y_train = data.iloc[:train_end, -1].values

    # # Validation set
    # X_val = data.iloc[train_end:val_end, :-1].values
    # y_val = data.iloc[train_end:val_end, -1].values

    # # Test set
    # X_test = data.iloc[val_end:, :-1].values
    # y_test = data.iloc[val_end:, -1].values
    # model = StockMLModelWithTraining()
    # model.compile(optimizer='adam', loss='mse')

    # Fit model

    # Predict next 30 values
    # n_predictions = 30
    # last_row = input_values[-WINDOW_SIZE:]
    # predictions = []
    # for i in range(n_predictions):
    #     # Predict next value based on last_row
    #     yhat = model.predict(np.array([last_row]))
    #     predictions.append(yhat[0, 0])  # Append predicted value to list of predictions
        
    #     # Update last_row with predicted value
    #     last_row = np.append(last_row[1:], yhat, axis=0)
    #     last_row[-1][-1] = yhat[0, 0]  # Replace last element of last row with predicted value
        
    #     # Add predicted value to input and target data
    #     input_values = np.concatenate([input_values, np.array([last_row[-1]])], axis=0)
    #     y = np.append(y, yhat[0, 0])
    #     X = []
    #     for i in range(WINDOW_SIZE, len(input_values)):
    #         X.append(input_values[i-WINDOW_SIZE:i])
    #     X = np.array(X)
        
    #     # Fit model with updated data
    #     model.fit(X, y, epochs=1, verbose=0)

    # print(predictions)

    # Predict next value
    # Demonstrate prediction
    # last_row = input_values[-n_steps:]
    # yhat = model.predict(np.array([last_row]))
    # print(yhat)

    # Output the input_values and target_value info and calculations
    # variance = np.var(target_value)
    # # variance = np.var(target_value['Close'])
    # print('The model input values for {stock} are as follows:\n{head}\n'.format(stock = STOCK_NAME, head = input_values.head()))
    # print('The model target values for {stock} are as follows:\n{head}\nvariance: {var}\n{sum}\n'.format(stock = STOCK_NAME, head = target_value.head(), var = variance, sum = target_value.describe()))

    # print('\nX_train:\n',X_train)
    # print('\nX_val:\n',X_val)
    # print('\ny_train:\n',y_train)
    # print('\ny_val:\n',y_val)
    # print('\nX_test:\n',X_test)
    # print('\ny_test:\n',y_test)
        
    # Initialize the machine learning model, then compile 
    
    # # convert into input/output
    # n_steps = 5
    # X, y = split_sequences(input_values, n_steps)
    # n_features = X.shape[2]
    # n_samples = X.shape[0]
    # print(X.shape, y.shape, n_features, n_samples)

    # model = StockMLModel()
    # X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    # y_train, y_val, y_test = preprocess(y_train, y_val, y_test)
    # # Compile the model
    # optimizer = optimizers.RMSprop()

    # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])

    # # Fit the model to the training data
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose = 0)

    # # demonstrate prediction
    # # Convert NumPy array to tensor
    # last_row_tensor = tf.convert_to_tensor(last_row, dtype=tf.float32)

    # yhat = model.predict(last_row_tensor, verbose=0)
    # print(yhat)

    # Save the model as a file
    # print(save_model(ml_model))


    # Instantiate the model with the training, validation, and test sets
    # arima_model = ARIMAModel(y_train, y_val, y_test)
    # arima_model.fit(ARIMA_P, ARIMA_D, ARIMA_Q)
    # arima__forecasting, conf_int = arima_model.forecast(days=DAYS)
    # print("The ARIMA Model Forecast:\n", arima__forecasting)


    # demonstrate prediction

    # ml_forecasting = ml_model.predict(pd.DataFrame(data = stock_model.X_val, columns=['Date', 'Open', 'High', 'Low', 'Volume']))
    # ml_forecasting = postprocess(ml_forecasting)

    # Instantiate the model with the training, validation, and test sets
    # arima_model = ARIMAModel(y_train, y_val, y_test)
    # arima_model.fit(ARIMA_P, ARIMA_D, ARIMA_Q)
    # arima__forecasting, conf_int = arima_model.forecast(days=DAYS)
    # print("The ARIMA Model Forecast:\n", arima__forecasting)

    # Evaluate the model
    # mse = mean_squared_error(y_val, ml_forecasting)
    # result = ''
    # if mse < (variance*0.66):
    #     result += 'Success, the MSE: {mse} is smaller than the Variance: {var}.'.format(mse= int(mse), var= int(variance))
    # else:
    #     result += 'Fail, the MSE: {mse} is not much smaller than the Variance: {var}.'.format(mse= int(mse), var= int(variance))
    # print(result)

    # Fit the model with p, d, and q values
    # mse_val, mse_test = arima_model.evaluate()
    # print(mse_val, mse_test)

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
def df_to_png_plot(stock_name):
    stock_df = pd.DataFrame()
    stock_df = get_stock_df('my_stock_list_quotes', stock_name, '{y}-{m}-{d}'.format(y=YY, m=MM, d=DD), TODAY)
    if stock_df is not None and stock_df.empty:
        stock_df = readSqliteTable('my_stock_list_quotes', stock_name, None)
    if stock_df is not None and stock_df.empty:
        print('Error!!! stock_df is empty.')
        return 
    image_dir = 'charts/'
    os.makedirs(image_dir, exist_ok=True)  # Create the directory if it doesn't exist

    image_path = os.path.join(image_dir, '{stock_name}.png'.format(stock_name=stock_name.lower()))
    if os.path.isfile(image_path):
        print("File exists for {s}.".format(s=stock_name))
    else:
        fig = create_figure(stock_df)
        fig.savefig(image_path, format='png')
        plt.close(fig)
        print("File does not exist for {s}.".format(s=stock_name))

def create_figure(df):
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="whitegrid")

    x = df['Date']
    y = df['Close']

    sns.lineplot(x=x, y=y, color="#917ed5")

    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.xlabel('Date (YYYY-MM-DD)', fontsize=12)

    return plt.gcf()

@app.route('/')
@app.route('/results', methods=("POST", "GET"))
def results():
    # Create Graphs for all the stocks
    stock_list_df = pd.DataFrame()
    for stock in STOCKS:
        # stock_data = df_to_png_plot(stock)
        # print(stock_data)
        # stock_list_df = get_stock_df('my_stock_list_quotes', stock, TODAY, TODAY)
        stock_list_df = get_stock_df('my_stock_list_quotes', stock, None, None)
    stock_list_data = stock_list_df.to_json()

    return render_template('results.html', title='Stock Forecasting', stocks = stock_list_data, today = TODAY)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80, debug = True)
    stock_list_df = pd.DataFrame()
    for stock in STOCKS:
        new_stock_df = get_stock_df('my_stock_list_quotes', stock, start_date, TODAY)
        stock_list_df = pd.concat([new_stock_df, stock_list_df])

    # Output the stock_list_data to a JSON file with indentation
    with open("stock_data.json", "w") as json_file:
        stock_list_df.to_json(json_file, orient="records", indent=4)

    print('\nXXX- the stock_list_df: -XXX')
    print(stock_list_df)