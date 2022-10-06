#!/bin/python3.9

from flask import Flask
app = Flask(__name__)
@app.route("/")
def index():
    return "Test App"

app.run(host="0.0.0.0", port=80)
