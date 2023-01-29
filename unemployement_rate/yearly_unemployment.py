#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:09:50 2022

@author: joanguich
"""


import pandas as pd



df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/unemployement_rate/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")


df.columns = ['Periods', "unemployment_rate", 'Codes']

df[['Year','Quarter']]=df.Periods.str.split('-',expand=True)


print('MEAN:')

df_mean = df.groupby(['Year']).mean()

print(df_mean)







print('STD:')

df_std = df.groupby(['Year']).std()

print(df_std)


print('SEM:')

df_sem = df_std


for i in range(47):
    
    df_sem['unemployment_rate'][i] *= (1/4)

    
print(df_sem)





ax = df_mean.plot.bar(rot=0, legend = False)


ax.set_title("Mean of the unemployment rate for each year (from 1975 to 2022)")
ax.set_xlabel("Years")
ax.set_ylabel("Unemployment rate")


ax.set_xticklabels(list(df_mean.index), rotation = 90)
