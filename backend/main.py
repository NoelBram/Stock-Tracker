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
from tensorflow.keras import layers

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

CHARTS_FOLDER = os.path.join('static', 'charts')

app = Flask(__name__)
# app.config['CHART_FOLDER'] = CHARTS_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = 'pk_913ba7d52f144907a92856b52ea0636e'
db = SQLAlchemy(app)

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

STOCK_LIST_DF = pd.DataFrame()
for stock in STOCKS:
    print('Here is Today: \n')
    print(TODAY)
    STOCK_LIST_DF = get_stock_df('my_stock_list_quotes', stock, TODAY, TODAY)

print('Here is a list of stock data of the most recent \'weekday\' before today.')
print(STOCK_LIST_DF.to_numpy())

# Dataframe to graph out the forcast in dropdown.
STOCK_DF = get_stock_df('my_stock_quotes', stock, '2021-12-23', TODAY)

test_size = 30
simulation_size = 10

class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        tf.compat.v1.disable_eager_execution()
        def lstm_cell(size_layer):
            return layers.LSTMCell(size_layer)

        rnn_cells = [lstm_cell(size_layer) for _ in range(num_layers)]
        stacked_lstm = layers.StackedRNNCells(rnn_cells)

        self.X = tf.compat.v1.placeholder(tf.float32, (None, None, size))
        self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
        drop = tf.nn.RNNCellDropoutWrapper(
            stacked_lstm, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.compat.v1.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        self.logits = layers.Dense(self.outputs[-1], output_size)
        self.cost = tf.compat.v1.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        
def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

# def forecast():
#     num_layers = 1
#     size_layer = 128
#     timestamp = 5
#     epoch = 300
#     dropout_rate = 0.8
#     future_day = test_size
#     learning_rate = 0.01
    
#     tf.compat.v1.reset_default_graph()
#     modelnn = Model(
#         learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
#     )
#     sess = tf.compat.v1.InteractiveSession()
#     sess.run(tf.compat.v1.global_variables_initializer())
#     date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

#     pbar = tqdm(range(epoch), desc = 'train loop')
#     for i in pbar:
#         init_value = np.zeros((1, num_layers * 2 * size_layer))
#         total_loss, total_acc = [], []
#         for k in range(0, df_train.shape[0] - 1, timestamp):
#             index = min(k + timestamp, df_train.shape[0] - 1)
#             batch_x = np.expand_dims(
#                 df_train.iloc[k : index, :].values, axis = 0
#             )
#             batch_y = df_train.iloc[k + 1 : index + 1, :].values
#             logits, last_state, _, loss = sess.run(
#                 [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
#                 feed_dict = {
#                     modelnn.X: batch_x,
#                     modelnn.Y: batch_y,
#                     modelnn.hidden_layer: init_value,
#                 },
#             )        
#             init_value = last_state
#             total_loss.append(loss)
#             total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
#         pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))
    
#     future_day = test_size

#     output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
#     output_predict[0] = df_train.iloc[0]
#     upper_b = (df_train.shape[0] // timestamp) * timestamp
#     init_value = np.zeros((1, num_layers * 2 * size_layer))

#     for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
#         out_logits, last_state = sess.run(
#             [modelnn.logits, modelnn.last_state],
#             feed_dict = {
#                 modelnn.X: np.expand_dims(
#                     df_train.iloc[k : k + timestamp], axis = 0
#                 ),
#                 modelnn.hidden_layer: init_value,
#             },
#         )
#         init_value = last_state
#         output_predict[k + 1 : k + timestamp + 1] = out_logits

#     if upper_b != df_train.shape[0]:
#         out_logits, last_state = sess.run(
#             [modelnn.logits, modelnn.last_state],
#             feed_dict = {
#                 modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0),
#                 modelnn.hidden_layer: init_value,
#             },
#         )
#         output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
#         future_day -= 1
#         date_ori.append(date_ori[-1] + timedelta(days = 1))

#     init_value = last_state
    
#     for i in range(future_day):
#         o = output_predict[-future_day - timestamp + i:-future_day + i]
#         out_logits, last_state = sess.run(
#             [modelnn.logits, modelnn.last_state],
#             feed_dict = {
#                 modelnn.X: np.expand_dims(o, axis = 0),
#                 modelnn.hidden_layer: init_value,
#             },
#         )
#         init_value = last_state
#         output_predict[-future_day + i] = out_logits[-1]
#         date_ori.append(date_ori[-1] + timedelta(days = 1))
    
#     output_predict = minmax.inverse_transform(output_predict)
#     deep_future = anchor(output_predict[:, 0], 0.4)
    
#     return deep_future

# @app.route('/results.png')
# def get_results():
#     results = []
#     for i in range(simulation_size):
#         print('simulation %d'%(i + 1))
#         results.append(forecast())
    
#     date_ori = pd.to_datetime(df.iloc[:, 1]).tolist()
#     for i in range(test_size):
#         date_ori.append(date_ori[-1] + timedelta(days = 1))
#     date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
#     print(date_ori[-5:])

#     accepted_results = []
#     for r in results:
#         if (np.array(r[-test_size:]) < np.min(df['Close'])).sum() == 0 and \
#         (np.array(r[-test_size:]) > np.max(df['Close']) * 2).sum() == 0:
#             accepted_results.append(r)
#     # len(accepted_results)

#     accuracies = [calculate_accuracy(df['Close'].values, r[:-test_size]) for r in accepted_results]

#     fig, ax = plt.figure(figsize = (15, 5))
#     fig.patch.set_facecolor('#E8E5DA')
#     for no, r in enumerate(accepted_results):
#         plt.plot(r, label = 'forecast %d'%(no + 1))
#     plt.plot(df['Close'], label = 'true trend', c = 'black')
#     plt.legend()
#     plt.title('Average accuracy: %.4f'%(np.mean(accuracies)))

#     x_range_future = np.arange(len(results[0]))
#     plt.xticks(x_range_future[::30], date_ori[::30])

#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
    
#     return Response(output.getvalue(), mimetype='image/png')

sio = SocketIO(app)

thread = None
thread_lock = Lock()

@app.route('/')
@app.route('/results', methods=("POST", "GET"))
def index():
    global thread
    return render_template('results.html', title='Stock Forcasting', stocks = STOCK_LIST_DF)


@app.route('/plot.png')
def plot_png():
    fig = create_figure(STOCK_DF)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(df):
    fig, ax = plt.subplots(figsize = (6,4))
    fig.patch.set_facecolor('#E8E5DA')

    x = df['Date']
    y = df['High']

    ax.bar(x, y, color = "#304C89")

    plt.ylabel('Stock Price (USD)', fontsize=8)
    plt.xlabel('Date (YYYY-MM-DD)', fontsize=8)

    return fig
# @app.route('/api/data')
# def data():
#     query = User.query

#     # search filter
#     search = request.args.get('search[value]')
#     if search:
#         query = query.filter(db.or_(
#             User.name.like(f'%{search}%'),
#             User.email.like(f'%{search}%')
#         ))
#     total_filtered = query.count()

#     # sorting
#     order = []
#     i = 0
#     while True:
#         col_index = request.args.get(f'order[{i}][column]')
#         if col_index is None:
#             break
#         col_name = request.args.get(f'columns[{col_index}][data]')
#         if col_name not in ['Symbol', 'Date', 'Date', 'High', 'Low', 'Close', 'Volume']:
#             col_name = 'Symbol'
#         descending = request.args.get(f'order[{i}][dir]') == 'desc'
#         col = getattr(User, col_name)
#         if descending:
#             col = col.desc()
#         order.append(col)
#         i += 1
#     if order:
#         query = query.order_by(*order)

#     # pagination
#     start = request.args.get('start', type=int)
#     length = request.args.get('length', type=int)
#     query = query.offset(start).limit(length)

#     # response
#     return {
#         'data': [user.to_dict() for user in query],
#         'recordsFiltered': total_filtered,
#         'recordsTotal': User.query.count(),
#         'draw': request.args.get('draw', type=int),
#     }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug = True)
