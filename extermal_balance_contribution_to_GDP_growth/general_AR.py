#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 10:42:15 2022

@author: joanguich
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg


df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/extermal_balance_contribution_to_GDP_growth/quarterly_values.csv", skiprows = [1, 2, 3], sep = ";")

df.columns = ['Periods', 'GDP_contribution', 'Codes']


df = df.drop(df.iloc[[288, 289, 290, 291, 292]].index)


df2 = pd.DataFrame({'date': pd.date_range('1950-4-01','2022-01-01', freq = 'QS')}).set_index('date')
aux = df["GDP_contribution"].tolist()


a = list((reversed(aux)))


df2['value'] = a


X = df2.values



train = X[:len(X) - 3]
test = X[len(X) - 3:]


model = AutoReg(train, lags = 11).fit()

print(model.summary())

