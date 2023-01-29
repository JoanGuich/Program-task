#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 20:09:44 2022

@author: joanguich
"""



import pandas as pd

import datetime

import matplotlib.pyplot as plt

import numpy as np




df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/GDP/annual_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns= ['Year', 'GDP', 'Codes']



df2 = pd.DataFrame({'date': pd.date_range('1949-1-01','2021-01-01', freq = 'YS')}).set_index('date')
aux = df["GDP"].tolist()


a = list((reversed(aux)))


df2['value'] = a



train = df2.iloc[:56]
test = df2.iloc[56:]





from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)



from keras.preprocessing.sequence import TimeseriesGenerator

#define generator

n_input = 2
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length = n_input, batch_size = 1)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#define model
model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape = (n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


model.fit(generator, epochs = 15)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)


last_train_batch = scaled_train[-2:]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))

model.predict(last_train_batch)

scaled_test[0]

test_predictions = []
    
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    #get the predixtion value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    #append the prediction into the array
    test_predictions.append(current_pred)
    
    #use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis = 1)
    
    
true_predictions = scaler.inverse_transform(test_predictions)   

test['Predictions'] = true_predictions


test.plot(figsize = (12,6))
    
    



