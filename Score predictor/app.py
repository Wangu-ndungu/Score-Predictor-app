# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:14:06 2022

@author: user
"""


import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['GET', 'POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction
           
    return render_template('index.html', prediction_text= 'Predicted final exam score {}'.format(output))
       
if __name__ == "__main__":
    app.run(debug=True)
