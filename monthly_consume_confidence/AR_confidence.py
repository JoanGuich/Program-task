#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:40:49 2022

@author: joanguich
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg





df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/monthly_consume_confidence/monthly_values.csv", skiprows = [1, 2, 3], sep = ";")


df.columns = ['Periods', "Household_confidence", 'Codes']



df2 = pd.DataFrame({'date': pd.date_range('1972-10-01','2022-06-01', freq = 'MS')}).set_index('date')


df2['value'] = list((reversed(df["Household_confidence"].tolist())))


X = df2.values



train = X[:len(X) - 16]
test = X[len(X) - 16:]


model = AutoReg(train, lags = 14).fit()

print(model.summary())


pred = model.predict(start=len(train), end = len(X)-1, dynamic = False)


from matplotlib import pyplot

pyplot.plot(pred)
pyplot.plot(test, color = 'red')
print(pred)


from math import sqrt
from sklearn.metrics import mean_squared_error

rmse = sqrt(mean_squared_error(test, pred))

print(test.mean())

print(rmse)

#print(test['value'].mean())


#Making future predictions


pred_future = model.predict(start=len(X), end = len(X)+13, dynamic = False)
print("The future prediction for the next months")
print(pred_future)
print('Number of Predictions Made: \t', len(pred_future))



index_future_dates = pd.date_range(start = '2022-06-01', end = '2023-07-01', freq = 'MS')
#print(index_future_dates)




pred = pd.DataFrame(pred_future, columns = ['values'])
pred.index = index_future_dates
#print(pred)





pred.plot(figsize = (12, 5), legend = True)

