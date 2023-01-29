#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:00:54 2022

@author: joanguich
"""

import pandas as pd

df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/french_pop/french_pop.csv", skiprows = [0, 1, 79, 80, 81, 82, 83, 84, 85])

df.columns = ['Year', 'Code',  'Population', 'Births', 'Deaths', 'Natural_increase', 'Net_Migration', 'Adjustment']




df2 = pd.DataFrame({'date': pd.date_range('1946-1-01','2021-01-01', freq = 'YS')}).set_index('date')
df2['value']  = df["Deaths"].tolist()

print(df2)




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

train = df2.iloc[:-3]
test = df2.iloc[-3:]
#print(train.shape, test.shape)




#model = ARIMA(train['value'], order = (5,2,1))
model = sm.tsa.arima.ARIMA(train['value'], order=(1,0,0))
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


model2 = sm.tsa.arima.ARIMA(df2['value'], order = (1,0,0))
model2 = model2.fit()
print(df2.tail())



#FOR FUTURE DATES

index_future_dates = pd.date_range(start = '2021-01-01', end = '2024-01-01', freq = 'YS')
#print(index_future_dates)

pred = model2.predict(start = len(df2), end = len(df2) + 3, typ = 'levels').rename('ARIMA Predictions')
pred.index = index_future_dates
#print(pred)

pred.plot(figsize = (12, 5), legend = True)













