# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:51:40 2023

@author: ssdim
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle

with open(r"C:\Users\ssdim\api_docker\model.pkl", "rb") as file:
    model = pickle.load(file)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'input' not in data:
        return jsonify({'error': 'Missing input data.'}), 400

    input_data = data['input']

    if not isinstance(input_data, list):
        return jsonify({'error': 'Invalid input format. Expected a list of values.'}), 400

    try:
        input_data = np.array(input_data, dtype=np.float32)
    except ValueError:
        return jsonify({'error': 'Invalid input data. Expected numeric values.'}), 400

    prediction = model.predict(np.expand_dims(input_data, axis=0))

    output = prediction.tolist()[0]

    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run()
