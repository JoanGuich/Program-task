#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:09:10 2022

@author: joanguich
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/french_pop/french_pop.csv", skiprows = [0, 1, 3, 4, 5, 6, 79, 80, 81, 82, 83, 84, 85])

df.columns = ['Year', 'Code',  'Population', 'Births', 'Deaths', 'Natural_increase', 'Net_Migration', 'Adjustment']


df2 = pd.read_csv("/Users/joanguich/Desktop/Citibeats/GDP/annual_values.csv", skiprows = [1, 2, 3, 76], sep = ';')

df2.columns = ['Year', "GDP", 'Codes']




aux = df2["GDP"].tolist()



y = list((reversed(aux)))

x = df.drop(['Year', 'Code', 'Adjustment'], axis = 1).values


#SPLIT THE DATASET IN TRAINING SET AND TEST SET

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 0)


#TRAIN THE MODEL ON THE TRAINING SET

from sklearn.linear_model import LinearRegression

ml = LinearRegression()
ml.fit(x_train, y_train)



#PREDICT THE TEST SET RESULTS

y_pred = ml.predict(x_test)
print(y_pred)



#EVALUATE THE MODEL

from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))


import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')



pred_y_df = pd.DataFrame({'Actual Value': y_test, 'Predicted Value' : y_pred, 'Difference' : y_test-y_pred, 'Error Percentage': y_test/(y_test-y_pred)*100})

print(pred_y_df)






