#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sns

# word cloud library
from wordcloud import WordCloud

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import Dataset
df = pd.read_csv('../input/countries of the world.csv')


# In[ ]:


#First 5 rows
df.head(5)


# In[ ]:


#summary of dataset
df.info()


# **Except Population, Area and GDP whole columns are string we need to change them to float. (Of course except Country and Region)**

# In[ ]:


#make columns names Lower Case
df.columns = map(str.lower, df.columns)
df.head()


# In[ ]:


#Replace ',' with '.'
#Convert columns from 'object' to 'float64' 
for column in df.columns[1:]:
    if df[column].astype(str).str.contains(',').any():
        df[column] = df[column].str.replace(',','.')
        df[column] = df[column].astype(float)


# In[ ]:


df.info()


# In[ ]:


# Sort the data according to GDP level
df = df.sort_values(by=['gdp ($ per capita)'],ascending=False)


# In[ ]:


# calculate the correlation matrix
corr = df.corr()

# plot the heatmap
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,ax=ax)

plt.show()


# * We can see the correlations of columns from heatmap.
# * According to this heatmap lets see how Birth Rate and Literacy change with GDP on the plot below

# In[ ]:


df = df.tail(1000)
trace1 = go.Scatter(
                    x = df['gdp ($ per capita)'],
                    y = df['birthrate'],
                    mode = "lines+markers",
                    name = "Birth Rate",
                    marker = dict(color = 'rgba(130, 70, 50, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df['gdp ($ per capita)'],
                    y = df['literacy (%)'],
                    mode = "lines+markers",
                    name = "Literacy",
                    marker = dict(color = 'rgba(120, 76, 150, 0.8)'),
                    text= df.country)
data = [trace1, trace2]
layout = dict(title = 'Birth Rate and Literacy vs GDP',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# * The plot above takes the last 100 GDP country  to make the analysis
# * As the plot has bumpy characteristic. We can say that Birth Rate goes down with higher GDP on contraray Literacy goes up

# In[ ]:


#Bar Chart
df_bar = df.head(5) 
trace1 = go.Bar(
    x = df_bar.country,
    y = df_bar.agriculture, 
    name = 'Agriculture',
    text = df_bar['gdp ($ per capita)']/1000,
    marker=dict(
        color='rgba(158, 202, 225, 0.7)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    )
)

trace2 = go.Bar(
   x = df_bar.country,
    y = df_bar.industry,
    name='Industry',
    text = df_bar['gdp ($ per capita)']/1000,
    marker=dict(
        color='rgba(78, 102, 75, 0.7)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    )
)

data = [trace1, trace2]
layout = go.Layout(title = "Agriculture and Industry value of first 5 High GDP Country",barmode='group')

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **Whole contries industry levels is higher than agriculture level**

# In[ ]:


#Pie Chart
df_pie = df.tail(5) 
fig = {
  "data": [
    {
      "values": df_pie.population,
      "labels": df_pie.country,
      "domain": {"x": [0, .58]},
      "name": "Population",
      "hoverinfo":"label+name",
      "hole": .4,
      "type": "pie"
    },
  ],
  "layout": {
        "title":"Population of 5 Low GDP Country",
        "annotations": [
            {
                "font": {
                    "size": 22
                },
                "showarrow": False,
                "text": "Population",
                "x": 0.20,
                "y": 0.5
            },
        ]
    }
}
iplot(fig)


# In[ ]:


#Bubble chart
df_bubble = df.head(100)
data = [
    {
        'x': df_bubble['gdp ($ per capita)'],
        'y': df_bubble['phones (per 1000)'],
        'mode': 'markers',
        'marker': {
            'color': df_bubble.deathrate,
            'size' : df_bubble.birthrate,
            'showscale': True
        },
        "text" :  df_bubble.country
    }
]
layout = go.Layout(
    title='GDP vs Phones (per 1000) with Industry Level',
    width=800,
    height=350,
    xaxis=dict(
        title='GDP per capita',
        gridcolor='rgb(255, 255, 255)',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Phones (per 1000)',
        gridcolor='rgb(255, 255, 255)',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# This plot tells us;
# * GDP between 20k and 30k have more phones than any other countries
# * Countries have nearly same Birthrate (size of the bubbles)
# * Lower GDP countries Deathrate is higher

# In[ ]:


df_word = df.head(50)
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(df_word.region))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# Word Cloud shows that 50 countries with high GDP is from Western Europe.

# In[ ]:




