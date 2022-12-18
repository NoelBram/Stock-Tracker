import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from flask import Flask, Response, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from scraper import *

CHARTS_FOLDER = os.path.join('static', 'charts')

app = Flask(__name__)
# app.config['CHART_FOLDER'] = CHARTS_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = 'pk_913ba7d52f144907a92856b52ea0636e'
db = SQLAlchemy(app)

STOCK_NAME = 'AAPL'
df = get_stock_df(STOCK_NAME, '2022-04-04', '2022-04-08')

minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index
df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
df_log = pd.DataFrame(df_log)
print(df_log.head())

test_size = 30
simulation_size = 10

df_train = df_log.iloc[:-test_size]
df_test = df_log.iloc[-test_size:]
nfeatures = 7
print('df.shape = {df}, df_train.shape = {df_train}, df_test.shape = {df_test}'.format(df = df.shape, df_train = df_train.shape, df_test = df_test.shape))

# class Model:
#     def __init__(
#         self,
#         learning_rate,
#         num_layers,
#         size,
#         size_layer,
#         output_size,
#         forget_bias = 0.1,
#     ):
#         tf.compat.v1.disable_eager_execution()
#         def lstm_cell(size_layer):
#             return layers.LSTMCell(size_layer)

#         rnn_cells = [lstm_cell(size_layer) for _ in range(num_layers)]
#         stacked_lstm = layers.StackedRNNCells(rnn_cells)

#         self.X = tf.compat.v1.placeholder(tf.float32, (None, None, size))
#         self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
#         drop = tf.nn.RNNCellDropoutWrapper(
#             stacked_lstm, output_keep_prob = forget_bias
#         )
#         self.hidden_layer = tf.compat.v1.placeholder(
#             tf.float32, (None, num_layers * 2 * size_layer)
#         )
#         self.outputs, self.last_state = tf.nn.dynamic_rnn(
#             drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
#         )
#         self.logits = layers.Dense(self.outputs[-1], output_size)
#         self.cost = tf.compat.v1.reduce_mean(tf.square(self.Y - self.logits))
#         self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
#             self.cost
#         )
        
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

num_layers = 1
size_layer = 128
timestamp = 5
epoch = 300
dropout_rate = 0.8
future_day = test_size
learning_rate = 0.01

def forecast():
    tf.compat.v1.reset_default_graph()
    modelnn = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    pbar = tqdm(range(epoch), desc = 'train loop')
    for i in pbar:
        init_value = np.zeros((1, num_layers * 2 * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(
                df_train.iloc[k : index, :].values, axis = 0
            )
            batch_y = df_train.iloc[k + 1 : index + 1, :].values
            logits, last_state, _, loss = sess.run(
                [modelnn.logits, modelnn.last_state, modelnn.optimizer, modelnn.cost],
                feed_dict = {
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                    modelnn.hidden_layer: init_value,
                },
            )        
            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
        pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))
    
    future_day = test_size

    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * 2 * size_layer))

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(
                    df_train.iloc[k : k + timestamp], axis = 0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days = 1))

    init_value = last_state
    
    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(o, axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    
    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.4)
    
    return deep_future

def get_results():
    results = []
    for i in range(simulation_size):
        print('simulation %d'%(i + 1))
        results.append(forecast())
    
    date_ori = pd.to_datetime(df.iloc[:, 1]).tolist()
    for i in range(test_size):
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
    print(date_ori[-5:])

    accepted_results = []
    for r in results:
        if (np.array(r[-test_size:]) < np.min(df['Close'])).sum() == 0 and \
        (np.array(r[-test_size:]) > np.max(df['Close']) * 2).sum() == 0:
            accepted_results.append(r)
    # len(accepted_results)

    accuracies = [calculate_accuracy(df['Close'].values, r[:-test_size]) for r in accepted_results]

    plt.figure(figsize = (15, 5))
    for no, r in enumerate(accepted_results):
        plt.plot(r, label = 'forecast %d'%(no + 1))
    plt.plot(df['Close'], label = 'true trend', c = 'black')
    plt.legend()
    plt.title('average accuracy: %.4f'%(np.mean(accuracies)))

    x_range_future = np.arange(len(results[0]))
    plt.xticks(x_range_future[::30], date_ori[::30])
    
    url = '/static/charts/{stock_name}.png'.format(stock_name = STOCK_NAME)

    plt.savefig(url, format='png')
    # full_filename = os.path.join(app.config['CHART_FOLDER'], url)
    return url

def get_df_graph(df):
    plt.figure(figsize = (15, 5))
    df.plot(kind = 'scatter',
        x = 'Date',
        y = 'Close',
        color = 'red')

    url = 'backend/static/charts/{stock_name}.png'.format(stock_name = STOCK_NAME.lower())

    plt.savefig(url, format='png')
    return url[7:]
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

@app.route('/')
def test_flask():
    stocks = df.to_numpy()
    # img_src = get_df_graph(df)
    img_src = None
    return render_template('results.html', title='Stock Price Forcasting App', stocks = stocks, img_src= img_src)
    # return render_template('results.html', title='IEX Trading')



if __name__ == '__main__':
    print(get_df_graph(df))
    app.run(host='0.0.0.0', port=80)
    # image_url = get_results()
