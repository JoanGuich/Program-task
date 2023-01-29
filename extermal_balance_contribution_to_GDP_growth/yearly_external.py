#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 20:19:54 2022

@author: joanguich
"""

import pandas as pd

import matplotlib.pyplot as plt


df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/extermal_balance_contribution_to_GDP_growth/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns = ['Periods', 'GDP_contribution', 'Codes']


#df['Periods'] = df['Periods'].str.replace(r'\D', ' ')

df['Periods'] = df['Periods'].str.replace('-', ' ')
df['Periods'] = df['Periods'].str.replace('T1', ' ')
df['Periods'] = df['Periods'].str.replace('T2', ' ')
df['Periods'] = df['Periods'].str.replace('T3', ' ')
df['Periods'] = df['Periods'].str.replace('T4', ' ')





print('MEAN:')

df_mean = df.groupby(['Periods']).mean()

print(df_mean)



print('STD:')

df_std = df.groupby(['Periods']).std()

print(df_std)


print('SEM:')


df_sem = df_std


for i in range(72):
    df_sem['GDP_contribution'][i+1] *= (1/4)


df_sem['GDP_contribution'] *= (1/3)

print(df_sem)


df_mean2 = df_mean.drop(df_mean.iloc[[0]].index)




ax = df_mean2.plot.bar(rot=0, legend = False)


ax.set_title("Mean of the external balance contribution to GDP growth yearly (from 1950 to 2022)")
ax.set_xlabel("Years")
ax.set_ylabel("GDP")
ax.set_xticklabels(list(df_mean2.index), rotation = 90)

#ax.bar_label(label_type='center')
#ax.set_yticks(df_mean2['GDP_contribution'], list(df_mean2.index))

"""
# Add legend

ax.legend()

# Auto space

#ax.tight_layout()

# Display plot

ax.show()
"""





