# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:27:23 2023
@author: ssdim
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from memory_profiler import profile
from sklearn.metrics import accuracy_score 
import pickle

# Load and preprocess the iris dataset
def load_preprocess_data():
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=65)

    return X_train, X_test, y_train, y_test
 
def train_and_evaluate_model1(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(units=1, activation='sigmoid', input_shape=(X_train.shape[1],)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=15, batch_size=25)

    loss, accuracy = model.evaluate(X_test, y_test)

    predictions = model.predict(X_test)
    print(predictions)

    return model
        
X_train, X_test, y_train, y_test = load_preprocess_data()
trained_model = train_and_evaluate_model1(X_train, X_test, y_train, y_test)

file_path = r"C:\Users\ssdim\api_docker\model.pkl" 
with open(file_path, "wb") as file:
    pickle.dump(trained_model, file)
