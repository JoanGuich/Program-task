#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 00:51:28 2022

@author: joanguich
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


import datetime





df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/extermal_balance_contribution_to_GDP_growth/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns = ['Periods', 'GDP_contribution', 'Codes']


df = df.drop(df.iloc[[288, 289, 290, 291, 292]].index)





df2 = pd.DataFrame({'date': pd.date_range('1950-4-01','2022-01-01', freq = 'QS')}).set_index('date')
aux = df["GDP_contribution"].tolist()


a = list((reversed(aux)))


df2['value'] = a


train = df2.iloc[:238]
test = df2.iloc[238:]




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)



from keras.preprocessing.sequence import TimeseriesGenerator

#define generator

n_input = 4
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length = n_input, batch_size = 1)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#define model
model = Sequential()
model.add(LSTM(150, activation = 'relu', input_shape = (n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


model.fit(generator, epochs = 75)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)


last_train_batch = scaled_train[-4:]
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
    
    















