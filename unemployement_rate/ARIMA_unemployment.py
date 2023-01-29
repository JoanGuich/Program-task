#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:19:10 2022

@author: joanguich
"""

import pandas as pd


import datetime



df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/unemployement_rate/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns = ['Periods', 'unemployment_rate', 'Codes']


df2 = pd.DataFrame({'date': pd.date_range('1975-1-01','2022-01-01', freq = 'QS')}).set_index('date')
aux = df["unemployment_rate"].tolist()


a = list((reversed(aux)))



df2['value'] = a





from statsmodels.tsa.stattools import adfuller


def ad_test(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print("1. ADF : " , dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : " , dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation : ", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)
    

#THE P-VALUE SHOULD BE, MORE OR LESS, SMALLER THAN 0.05 for being stationary
ad_test(df2['value'])




from pmdarima import auto_arima
#Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


stepwise_fit = auto_arima(df2['value'], trace = True, suppress_warnings = True)

#print(stepwise_fit.summary())



#from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

#print(df2.shape)

train = df2.iloc[:-11]
test = df2.iloc[-11:]
#print(train.shape, test.shape)




#model = ARIMA(train['value'], order = (5,2,1))
model = sm.tsa.arima.ARIMA(train['value'], order=(5,2,1))
model = model.fit()
#print(model.summary())


start = len(train)
end = len(train) + len(test) - 1
pred = model.predict(start=start, end=end, typ='levels')
pred.index = df2.index[start:end+1]
#print(pred)


pred.plot(legend=True)
test['value'].plot(legend = True)



print(test['value'].mean())

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(pred, test['value']))
print(rmse)


model2 = sm.tsa.arima.ARIMA(df2['value'], order = (5,2,1))
model2 = model2.fit()
print(df2.tail())



#FOR FUTURE DATES

index_future_dates = pd.date_range(start = '2022-01-01', end = '2023-07-01', freq = 'QS')
#print(index_future_dates)

pred = model2.predict(start = len(df2), end = len(df2) + 6, typ = 'levels').rename('ARIMA Predictions')
pred.index = index_future_dates
print('PREDICCIONS:')
print(pred)

pred.plot(figsize = (12, 5), legend = True)













