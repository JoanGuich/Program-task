#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:51:45 2022

@author: joanguich
"""

import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
from prophet import Prophet

df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/extermal_balance_contribution_to_GDP_growth/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns = ['Periods', 'GDP_contribution', 'Codes']


df = df.drop(df.iloc[[288, 289, 290, 291, 292]].index)







df2 = pd.DataFrame({'date': pd.date_range('1950-4-01','2022-01-01', freq = 'QS')}).set_index('date')


period = df["Periods"].tolist()

b = list((reversed(period)))


df2['Date'] = b

df2['Date'] = df2['Date'].str.replace('-', ' ')


df2['Date'] = df2['Date'].str.replace('T1', 'T1')
df2['Date'] = df2['Date'].str.replace('T3', 'T7')
df2['Date'] = df2['Date'].str.replace('T4', 'T10')
df2['Date'] = df2['Date'].str.replace('T2', 'T4')


aux = df["GDP_contribution"].tolist()


a = list((reversed(aux)))


df2['value'] = a



df2.columns = ['ds', 'y']

df2['ds'] = pd.to_datetime(df2['ds'])




train = df2.iloc[:len(df2) - 50]
test = df2.iloc[len(df2) - 50:]


m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods = 4)
forecast = m.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

print(test.tail())


plot_plotly(m, forecast)

plot_components_plotly(m, forecast)


#Evaluate your model

from statsmodels.tools.eval_measures import rmse

predictions = forecast.iloc[-50:]['yhat']

print("Root Mean Squared Error between actual and predicted values: ", rmse(predictions, test['y']))
print("Mean Value of Test Dataset: ", test['y'].mean())













