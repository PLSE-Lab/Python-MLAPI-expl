#!/usr/bin/env python
# coding: utf-8

# This is my 2nd kernel ever. Learning as I go.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[2]:


def col_trim(df, *cols):
    """Takes a DataFrame, df, and any number of its columns, *cols, as args and trims the last word of each value.
    To avoid repitition in words. For example: Blue Hair, Brown Hair, Bald becomes Blue, Brown, Bald"""
    def f(x):
        y = str(x)
        y = y.split()
        if len(y) > 1:
            return " ".join(str(x).split()[:-1])
        return x
    for col in cols:
        df[col] = df[col].map(f)


# In[3]:


dc = pd.read_csv('../input/dc-wikia-data.csv', index_col='name')
dc = dc.drop(['GSM', 'urlslug', 'page_id'], 1)
col_trim(dc, 'EYE', 'HAIR', 'SEX', 'ID', 'ALIGN', 'ALIVE')
dc.head()


# In[27]:


marvel = pd.read_csv('../input/marvel-wikia-data.csv', index_col="name")
marvel = marvel.drop(['GSM', 'urlslug', 'page_id'], 1)
col_trim(marvel, 'EYE', 'HAIR', 'SEX', 'ID', 'ALIGN', 'ALIVE')
marvel.head()


# In[5]:


df_whole = marvel.append(dc, sort=True)


# ## <center> Comparison of DC and Marvel Comics</center>
# 

# In[6]:


eye_color_count_m = marvel['EYE'].value_counts()
eye_color_count_d = dc['EYE'].value_counts()

trace1 = go.Bar(
    x=eye_color_count_m.index,
    y=eye_color_count_m.values,
    name='Marvel'
)
trace2 = go.Bar(
    x=eye_color_count_d.index,
    y=eye_color_count_d.values,
    name='DC'
)
data = [trace1, trace2]

layout = go.Layout(
    barmode='group',
    title='Eye Color Comparisons Between DC and Marvel'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# In both DC and Marvel, blue eyes are the most common eye color to have among superheroes. Brown is second. In reality, blue eyes only account for 8% of the world's population while brown eyes account for an estimated 70-90%: <a href="https://www.aclens.com/Most-Common-Eye-Color">Eye Color Guide</a>.

# In[7]:


hair_count_m = marvel['HAIR'].value_counts()
hair_count_d = dc['HAIR'].value_counts()

trace1 = go.Bar(
    x=hair_count_m.index,
    y=hair_count_m.values,
    name='Marvel'
)
trace2 = go.Bar(
    x=hair_count_d.index,
    y=hair_count_d.values,
    name='DC'
)
data = [trace1, trace2]

layout = go.Layout(
    barmode='group',
    title='Hair Color Comparisons Between DC and Marvel'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# While superheroes have an abnormally high rate of blue eyes, they most commonly have black and brown hair. 

# In[21]:


vc = df_whole.ALIGN.value_counts()

colors = ['pink', 'aqua', 'gold', 'lightblue']

trace = go.Pie(
    labels  = vc.index,
    values  = vc.values,
    name    = 'Alignment',
    hole    = 0.3,
    marker  = dict(colors=colors)
)

data = [trace]

layout = go.Layout(
    title="Characters By Alignment"
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# More bad characters than good. Makes sense. Reformed is so small that it's not registering. The actual value for reformed is .015107%.
