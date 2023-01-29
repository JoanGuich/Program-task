#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 20:16:57 2022

@author: joanguich
"""

import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
from prophet import Prophet

df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/GDP/annual_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns= ['Year', 'GDP', 'Codes']



df = df.drop([0,1,72])




df2 = pd.DataFrame({'date': pd.date_range('1950-1-01','2019-01-01', freq = 'YS')}).set_index('date')

year = df["Year"].tolist()

b = list((reversed(year)))




df2['Period'] = b

df2['Period'] = df2['Period'].astype(str) + '-01-01'


aux = df["GDP"].tolist()


a = list((reversed(aux)))


df2['value'] = a





df2.columns = ['ds', 'y']

df2['ds'] = pd.to_datetime(df2['ds'])






train = df2.iloc[:len(df2) - 10]
test = df2.iloc[len(df2) - 10:]


m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods = 1)
forecast = m.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

print(test.tail())


plot_plotly(m, forecast)

plot_components_plotly(m, forecast)


#Evaluate your model

from statsmodels.tools.eval_measures import rmse

predictions = forecast.iloc[-10:]['yhat']

print("Root Mean Squared Error between actual and predicted values: ", rmse(predictions, test['y']))
print("Mean Value of Test Dataset: ", test['y'].mean())




