#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:22:52 2022

@author: joanguich
"""

import pandas as pd



df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/monthly_consume_confidence/monthly_values.csv", skiprows = [1, 2, 3], sep = ";")


df.columns = ['Periods', "Household_confidence", 'Codes']

df[['Year','Month']]=df.Periods.str.split('-',expand=True)




print('MEAN:')

df_mean = df.groupby(['Month']).mean()

print(df_mean)



print('STD:')

df_std = df.groupby(['Month']).std()

print(df_std)



print('SEM:')


df_sem = df_std


for i in range(6):
    df_sem['Household_confidence'][i] *= (1/50)


for i in range(3):
    df_sem['Household_confidence'][i+6] *= (1/49)

for i in range(3):
    df_sem['Household_confidence'][i+9] *= (1/50)


print(df_sem)



ax = df_mean.plot.bar(rot=0, legend = False)


ax.set_title("Mean of the households\' confidence for each month (from 1972 to 2022)")
ax.set_xlabel("Years")
ax.set_ylabel("Indicator of households\' confidence")

ax.set_xticklabels(list(df_mean.index), rotation = 90)


#df['Periods'] = df['Periods'].str.replace('-', ' ')





