from flask import Flask,request,render_template, redirect
import pandas as pd
import numpy as np
import pickle
application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/testing')
def testing():
    return render_template('home.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        data = pd.read_csv(file)
        processed_data = data_processing.transform(data)
        predictions = model.predict(processed_data)
        return render_template('index.html', results=predictions)

if __name__=='__main__':
    application.run()
    