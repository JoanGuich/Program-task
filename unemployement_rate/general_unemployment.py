#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:47:00 2022

@author: joanguich
"""

import pandas as pd


import math

import matplotlib.pyplot as plt



import datetime





df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/unemployement_rate/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns = ['Periods', 'unemployment_rate', 'Codes']




print('MEAN:')

df_mean = df['unemployment_rate'].mean()

print(df_mean)



print('STD:')

df_std = df['unemployment_rate'].std()

print(df_std)



print('SEM:')

df_sem = df_std * (1/math.sqrt(len(df) - 5))

print(df_sem)





df2 = pd.DataFrame({'date': pd.date_range('1975-1-01','2022-01-01', freq = 'QS')}).set_index('date')
aux = df["unemployment_rate"].tolist()


a = list((reversed(aux)))



df2['value'] = a



from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df2)

result.plot()



from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df2)




from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(df2)





from statsmodels.tsa.stattools import adfuller

adf, pval, usedlag, nobs, crit_vals, icbest = adfuller(df2.values)

print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)




# create xticks
xticks = pd.date_range(datetime.datetime(1975,1,1), datetime.datetime(2022,1,1), freq='YS')


# plot
fig, ax = plt.subplots(figsize=(12,8))
df2['value'].plot(ax=ax,xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%Y') for x in xticks]);
ax.set_ylabel("Unemployment rate")
ax.set_xlabel("Year")
ax.set_title("Time series of the unemployment rate by quarters (from 1975 to 2022)")
plt.xticks(rotation=90);
plt.show()

