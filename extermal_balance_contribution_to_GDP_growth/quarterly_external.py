#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:45:25 2022

@author: joanguich
"""

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import numpy as np





df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/extermal_balance_contribution_to_GDP_growth/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns = ['Periods', 'GDP_contribution', 'Codes']



df.loc[df['Periods'].str.contains('T1'), 'Periods'] = 'T1'
df.loc[df['Periods'].str.contains('T2'), 'Periods'] = 'T2'
df.loc[df['Periods'].str.contains('T3'), 'Periods'] = 'T3'
df.loc[df['Periods'].str.contains('T4'), 'Periods'] = 'T4'



print('MEAN:')

df_mean = df.groupby(['Periods']).mean()

print(df_mean)



print('STD:')

df_std = df.groupby(['Periods']).std()

print(df_std)


print('SEM:')

df_sem = df_std.T 


df_sem['T1'] = df_sem['T1'].apply(lambda x: x*(1/df.groupby('Periods').size()['T1']))
df_sem['T2'] = df_sem['T1'].apply(lambda x: x*(1/df.groupby('Periods').size()['T2']))
df_sem['T3'] = df_sem['T3'].apply(lambda x: x*(1/df.groupby('Periods').size()['T3']))
df_sem['T4'] = df_sem['T4'].apply(lambda x: x*(1/df.groupby('Periods').size()['T4']))

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


ax.set_ylabel("GDP")
ax.set_xlabel("Quarters")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title("Mean of the external balance contribution to GDP growth for each quarter (from 1950 to 2022)")
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()








