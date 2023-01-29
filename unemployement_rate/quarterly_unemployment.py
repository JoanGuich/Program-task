#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:55:36 2022

@author: joanguich
"""


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/unemployement_rate/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")


df.columns = ['Periods', "unemployment_rate", 'Codes']

df[['Year','Quarter']]=df.Periods.str.split('-',expand=True)


print('MEAN:')

df_mean = df.groupby(['Quarter']).mean()

print(df_mean)



print('STD:')

df_std = df.groupby(['Quarter']).std()

print(df_std)


print('SEM:')

df_sem = df_std.T 


df_sem['T1'] = df_sem['T1'].apply(lambda x: x*(1/df.groupby('Quarter').size()['T1']))
df_sem['T2'] = df_sem['T1'].apply(lambda x: x*(1/df.groupby('Quarter').size()['T2']))
df_sem['T3'] = df_sem['T3'].apply(lambda x: x*(1/df.groupby('Quarter').size()['T3']))
df_sem['T4'] = df_sem['T4'].apply(lambda x: x*(1/df.groupby('Quarter').size()['T4']))

df_sem = df_sem.T

print(df_sem)


mean_np = df_mean.to_numpy()
std_np = df_std.to_numpy()
sem_np = df_sem.to_numpy()



# Define labels, positions, bar heights and error bar heights
labels = ['T1', 'T2', 'T3', 'T4']
x_pos = np.arange(len(labels))
CTEs = [mean_np[0], mean_np[1], mean_np[2], mean_np[3]]
error = [sem_np[0], sem_np[1], sem_np[2], sem_np[3]]

CTEs = [float(i) for i in CTEs]
error = [float(i) for i in error]


# Build the plot
fig, ax = plt.subplots()



ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)


ax.set_ylabel("Unemployment rate")
ax.set_xlabel("Quarters")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title("Mean of the unemployment rate for each quarter (from 1975 to 2022)")
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()









