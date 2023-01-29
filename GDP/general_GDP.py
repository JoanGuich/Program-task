#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:52:01 2022

@author: joanguich
"""


import pandas as pd

import datetime

import matplotlib.pyplot as plt




df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/GDP/annual_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns= ['Year', 'GDP', 'Codes']



df2 = pd.DataFrame({'date': pd.date_range('1949-1-01','2021-01-01', freq = 'YS')}).set_index('date')
aux = df["GDP"].tolist()


a = list((reversed(aux)))


df2['value'] = a


"""
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df2)

result.plot()



from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df2)




from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(df2)

"""


from statsmodels.tsa.stattools import adfuller

adf, pval, usedlag, nobs, crit_vals, icbest = adfuller(df2.values)

print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)




# create xticks
xticks = pd.date_range(datetime.datetime(1949,1,1), datetime.datetime(2021,1,1), freq='YS')


# plot
fig, ax = plt.subplots(figsize=(12,8))
df2['value'].plot(ax=ax,xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%Y') for x in xticks]);
ax.set_ylabel("GDP")
ax.set_xlabel("Years")
ax.set_title("Time series of GDP for each year (from 1949 to 2022)")
plt.xticks(rotation=90);
plt.show()

