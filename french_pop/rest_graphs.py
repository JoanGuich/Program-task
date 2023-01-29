#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:47:41 2022

@author: joanguich
"""


import pandas as pd

import numpy as np



df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/french_pop/french_pop.csv", skiprows = [0, 1, 79, 80, 81, 82, 83, 84, 85])

df.columns = ['Year', 'Code',  'Population', 'Births', 'Deaths', 'Natural_increase', 'Net_Migration', 'Adjustment']




df2 = pd.DataFrame({'date': pd.date_range('1946-01-01','2021-01-01', freq = 'YS')}).set_index('date')

aux = df["Births"].tolist()




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


import datetime
import matplotlib.pyplot as plt

# create xticks
xticks = pd.date_range(datetime.datetime(1946,1,1), datetime.datetime(2021,1,1), freq='YS')


# plot
fig, ax = plt.subplots(figsize=(12,8))
df2['value'].plot(ax=ax,xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%Y') for x in xticks]);
ax.set_ylabel("GDP")
ax.set_xlabel("Years")
ax.set_title("Time series of Births for each year (from 1946 to 2021)")
plt.xticks(rotation=90);
plt.show()







df2 = pd.DataFrame({'date': pd.date_range('1946-01-01','2021-01-01', freq = 'YS')}).set_index('date')

aux = df["Deaths"].tolist()




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


import datetime
import matplotlib.pyplot as plt

# create xticks
xticks = pd.date_range(datetime.datetime(1946,1,1), datetime.datetime(2021,1,1), freq='YS')


# plot
fig, ax = plt.subplots(figsize=(12,8))
df2['value'].plot(ax=ax,xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%Y') for x in xticks]);
ax.set_ylabel("GDP")
ax.set_xlabel("Years")
ax.set_title("Time series of Deaths for each year (from 1946 to 2021)")
plt.xticks(rotation=90);
plt.show()










"""

df2 = pd.DataFrame(df, columns =['Year','Natural_increase'])


ax = df2.plot.bar(rot=0, legend = False)


ax.set_title("Natural increase of population in France each year (from 1946 to 2021)")
ax.set_xlabel("Year")
ax.set_ylabel("Natural increase")

ax.set_xticklabels(list(df2['Year']), rotation = 90)








df2 = pd.DataFrame(df, columns =['Year','Net_Migration'])


ax = df2.plot.bar(rot=0, legend = False)


ax.set_title("Net migration in France each year (from 1946 to 2021)")
ax.set_xlabel("Year")
ax.set_ylabel("Net Migration")

ax.set_xticklabels(list(df2['Year']), rotation = 90)



