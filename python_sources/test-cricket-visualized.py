#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random
import plotly.graph_objects as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# In[ ]:


data = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv', encoding='ISO-8859-1')


# In[ ]:


def row(player):
    if '*' in player['HS']:
        return 1
    else:
        return 0

data['hs_not_out'] = data.apply(row, axis=1)
data['HS'] = data['HS'].str.replace('*', '')
data['HS'] = data['HS'].str.replace('-', '0')
data['Mat'] = data['Mat'].replace('-', '0')
data['Inn'] = data['Inn'].str.replace('-', '0')
data['NO'] = data['NO'].str.replace('-', '0')
data['Runs'] = data['Runs'].str.replace('-', '0')
data['100'] = data['100'].str.replace('-', '0')
data['50'] = data['50'].str.replace('-', '0')
data['0'] = data['0'].str.replace('-', '0')
data['Avg'] = data['Avg'].str.replace('-', '0')
data['Inn'] = data['Inn'].astype('int32')
data['NO'] = data['NO'].astype('int32')
data['Runs'] = data['Runs'].astype('int32')
data['HS'] = data['HS'].astype('int32')
data['Avg'] = data['Avg'].astype('float32')
data['100'] = data['100'].astype('int32')
data['50'] = data['50'].astype('int32')
data['0'] = data['0'].astype('int32')
data['Country'] = data.Player.apply(lambda st: st[st.find("(")+1:st.find(")")].replace("ICC/",''))
data['Player'] = data['Player'].apply(lambda x: x.split("(")[0])
data['Span_Years'] = data['Span'].apply(lambda x: int(x.split("-")[1])-int(x.split("-")[0])).apply(lambda x: "5" if x<=5 else "10" if x<=10 else "15" if x <= 15 else "15+")
data['Span_Yrs'] = data['Span'].apply(lambda x: int(x.split("-")[1])-int(x.split("-")[0]))
data_filter = data[(data.Inn>=50) & (data.Runs >= 5000)]
# data_filter.shape


# In[ ]:


runs = []
for i in data_filter['Runs']:
    runs.append(i)
hover_text = []
for index, row in data_filter.iterrows():
    hover_text.append(('Player: {}<br>'+         'Country: {}<br>'+        'Runs: {}<br>'+        'Highest Score: {}<br>').format(row['Player'], row['Country'],row['Runs'], row['HS']))

max_ = data_filter['HS'].max()
fig = go.Figure(data=[go.Scatter(
    x=data_filter['Inn'], y=runs,
    text=hover_text,
    mode='markers',
    marker=dict(
        color=data_filter['Span_Yrs'],
        size=20*data_filter['HS']/max_,
        showscale = True
    )
)])

fig.update_layout(
    title='Runs v. Innings',
    xaxis=dict(
        title='Innings'
    ),
    yaxis=dict(
        title='Runs'
    )
)

fig.show()


# In[ ]:


hover_text = []
for index, row in data_filter.iterrows():
    hover_text.append(('Player: {}<br>'+         'Country: {}<br>'+        'Tons Per Duck: {}<br>'+        'Not Outs: {}<br>').format(row['Player'], row['Country'],row['100']/row['0'] ,row['NO']))
fig = go.Figure(data=[go.Scatter(
    x=data_filter['Inn'], y=data_filter['100']/data_filter['0'],
    text=hover_text,
    mode='markers',
    marker=dict(
        size=data_filter['NO'],
    )
)])

fig.update_layout(
    title='Tons/Ducks v. Innings',
    xaxis=dict(
        title='Innings'
    ),
    yaxis=dict(
        title='Tons/Ducks'
    )
)

fig.show()

