from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from scraper import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = 'pk_913ba7d52f144907a92856b52ea0636e'
db = SQLAlchemy(app)

class User(db.Model):
    Symbol = db.Column(db.String(7), primary_key=True)
    Date = db.Column(db.String(10), )
    Open = db.Column(db.Integer, index=True)
    High = db.Column(db.Integer, index=True)
    Low = db.Column(db.Integer)
    Close = db.Column(db.Integer)
    Volume = db.Column(db.Integer)

    # def stocks_dict(self):
    #     return {
    #         'Symbol': self.Symbol,
    #         'Date': self.Date,
    #         'Open': self.Open,
    #         'High': self.High,
    #         'Low': self.Low,
    #         'Close': self.Close,
    #         'Volume': self.Volume,
    #     }
    

db.create_all()


@app.route('/')
def index():
    Query = load_data()
    return render_template('results.html', title='IEX Trading', stocks = Query )


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
    app.run(host="0.0.0.0", port=80)
    # Query = User.query
    # Query = load_data()
    # print(Query)
