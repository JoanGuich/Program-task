#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:43:57 2022

@author: joanguich
"""



import pandas as pd





df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/monthly_consume_confidence/monthly_values.csv", skiprows = [1, 2, 3], sep = ";")


df.columns = ['Periods', "Household_confidence", 'Codes']

df[['Year','Month']]=df.Periods.str.split('-',expand=True)




print('MEAN:')

df_mean = df.groupby(['Year']).mean()

print(df_mean)



print('STD:')

df_std = df.groupby(['Year']).std()

print(df_std)



print('SEM:')


df_sem = df_std


for i in range(49):
    df_sem['Household_confidence'][i+1] *= (1/12)


df_sem['Household_confidence'][0] *= (1/3)
df_sem['Household_confidence'][50] *= (1/6)




print(df_sem)



ax = df_mean.plot.bar(rot=0, legend = False)


ax.set_title("Mean of the households\' confidence by years (from 1972 to 2022)")
ax.set_xlabel("Years")
ax.set_ylabel("Indicator of households\' confidence")

ax.set_xticklabels(list(df_mean.index), rotation = 90)


#df['Periods'] = df['Periods'].str.replace('-', ' ')














