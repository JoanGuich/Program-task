#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:30:17 2022

@author: joanguich
"""

import pandas as pd

import numpy as np


from sklearn import decomposition #PCA
import plotly.express as px

from sklearn.decomposition import PCA






userows = np.zeros((25,), dtype=int)

for i in range(25):
    userows[i] = i+1
    

df = pd.read_csv("/Users/joanguich/Desktop/Citibeats/extermal_balance_contribution_to_GDP_growth/yearly_GDP.csv", sep = ".", skiprows=lambda x: x in userows)



df2 = pd.read_csv("/Users/joanguich/Desktop/Citibeats/monthly_consume_confidence/year_confidence.csv", skiprows = [1, 2, 3])


df['Confidence'] = df2['Household_confidence']



df2 = pd.read_csv("/Users/joanguich/Desktop/Citibeats/unemployement_rate/year_unemployment.csv")


df['Unemployment'] = df2['unemployment_rate']




userows3 = np.zeros((29,), dtype=int)

userows3[0] = 1
userows3[1] = 2
userows3[2] = 3

for i in range(26):
    userows3[i+3] = i+51



df2 = pd.read_csv("/Users/joanguich/Desktop/Citibeats/GDP/annual_values.csv", skiprows=lambda x: x in userows3, sep = ";")

df2.columns= ['Year', 'GDP', 'Codes']

aux = df2['GDP'].tolist()


a = list((reversed(aux)))

df2['GDP'] = a



df['GDP'] = df2['GDP']





userows2 = np.zeros((31,), dtype=int)

for i in range(31):
    userows2[i] = i+1

df2 = pd.read_csv("/Users/joanguich/Desktop/Citibeats/french_pop/french_pop.csv", skiprows=lambda x: x in userows2)

df2 = df2.drop([48, 49, 50, 51, 52, 53])


df2.columns = ['Year', 'Code',  'Population', 'Births', 'Deaths', 'Natural_increase', 'Net_Migration', 'Adjustment']



df['Population'] = df2['Population']
df['Births'] = df2['Births']
df['Deaths'] = df2['Deaths']
df['Net_Migration'] = df2['Net_Migration']

df = df.drop(47)




convert_dict = {'Births': int, 'Deaths': int, 'Net_Migration': int}
 
df = df.astype(convert_dict)






#STANDARIZE DATA

df['GDP_contribution'] =( df['GDP_contribution'] - df['GDP_contribution'].mean() ) / df['GDP_contribution'].std()
df['Confidence'] =( df['Confidence'] - df['Confidence'].mean() ) / df['Confidence'].std()
df['Unemployment'] =( df['Unemployment'] - df['Unemployment'].mean() ) / df['Unemployment'].std()
df['GDP'] =( df['GDP'] - df['GDP'].mean() ) / df['GDP'].std()
df['Population'] =( df['Population'] - df['Population'].mean() ) / df['Population'].std()
df['Births'] =( df['Births'] - df['Births'].mean() ) / df['Births'].std()
df['Deaths'] =( df['Deaths'] - df['Deaths'].mean() ) / df['Deaths'].std()
df['Net_Migration'] =( df['Net_Migration'] - df['Net_Migration'].mean() ) / df['Net_Migration'].std()






X = np.column_stack((df['GDP_contribution'], df['Confidence'], df['Unemployment'], df['GDP'], df['Population'], df['Births'], df['Deaths'], df['Net_Migration']))
Y = df['Periods']


features = ['GDP_contribution', 'Confidence', 'Unemployment', 'GDP', 'Population', 'Births', 'Deaths', 'Net_Migration']


pca = decomposition.PCA(n_components=8)
pca.fit(X)

scores = pca.transform(X)

scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
print(scores_df)






labels = pd.DataFrame(Y, columns=['Periods'])

df_scores = pd.concat([scores_df, labels], axis=1)


loadings = pca.components_.T
df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2','PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], index=features)
print(df_loadings.to_string())



explained_variance = pca.explained_variance_ratio_



print(explained_variance)





explained_variance = np.insert(explained_variance, 0, 0)

cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

pc_df = pd.DataFrame(['','PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], columns=['PC'])
explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])

df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
print(df_explained_variance)







import plotly.io as pio
pio.renderers.default='browser'


import matplotlib.pyplot as plt



#HERE WE JUST PLOT THE FILTERED VALUES IN ALL THE PLANES WHOSE AXES ARE TWO OF THREE FIRST PRINCIPAL COMPONENTS
#THERE IS GONNA BE A TOTAL OF 6 PLOTS, WITH THE EXPLAINED VARIANCE REPRESENTED BY PERCENTAGES (%)


pca = PCA()
components = pca.fit_transform(df[features])
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}



fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(3),
    color=df["Periods"] #HERE YOU CAN CHANGE THE LABEL MAPPED TO THE COLOR FUNCTION
)
fig.update_traces(diagonal_visible=False)
fig.show()





#WE PLOT THE FILTERED VALUES INTO THE PLANE WITH THE FIRST AND SECOND PRINCIPAL COMPONENTS AS AXES,
#BUT INCLUDING LINES IN THE CENTER REPRESENTING THE VARIABLES DIRECTIONS OVER PCs (LOADINGS)


pca = PCA(n_components=2)
components = pca.fit_transform(df[features])

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig = px.scatter(components, x=0, y=1, color=df['Periods'])

for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )



fig.show()





#WE CREATE AN HISTOGRAM FOR THE EXPLAINED VARIACNE OF EVERY VARIABLE, AND ALSO THE CUMULATIVE



pca = PCA()

X_train_pca = pca.fit_transform(df[features])
#
# Determine explained variance using explained_variance_ration_ attribute
#
exp_var_pca = pca.explained_variance_ratio_
#
# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
#
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
#
# Create the visualization plot
#
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()






