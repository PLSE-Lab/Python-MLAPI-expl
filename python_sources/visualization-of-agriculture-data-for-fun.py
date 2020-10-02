#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


import plotly.graph_objs as go


# In[ ]:


df = pd.read_csv('/kaggle/input/agricultural-raw-material-prices-19902020/agricultural_raw_material.csv')


# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


percentages=['Coarse wool price % Change',
       'Copra price % Change', 
       'Cotton price % Change', 'Fine wool price % Change',
       'Hard log price % Change', 
       'Hard sawnwood price % Change', 'Hide price % change',
       'Plywood price % Change', 
       'Rubber price % Change', 'Softlog price % Change',
       'Soft sawnwood price % Change',
       'Wood pulp price % Change']


# Correlation Matrix

# In[ ]:


corrmat = df.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True, annot = True)
plt.show()


# In[ ]:


df=df.replace('-', '0')


# In[ ]:


for i in percentages:
    df[i]=df[i].str.replace('%', '')
    df[i]=df[i].astype('float')


# In[ ]:


df.describe()


# In[ ]:


prices=['Coarse wool Price',
       'Copra Price', 'Cotton Price',
       'Fine wool Price',
       'Hard log Price', 'Hard sawnwood Price',
       'Hide Price', 
       'Plywood Price', 'Rubber Price',
       'Softlog Price', 
       'Soft sawnwood Price', 
       'Wood pulp Price']


# In[ ]:


colors=['#b84949', '#ff6f00', '#ffbb00', '#9dff00', '#329906', '#439c55', '#67c79e', '#00a1db', '#002254', '#5313c2', '#c40fdb', '#e354aa']


# Univariate Analysis

# In[ ]:


#univariate analysis
import plotly.express as px
x=0
for i in prices:
    #df = px.data.tips()
    fig = px.histogram(df, x=i, nbins=100, opacity=0.8,
                   color_discrete_sequence=[colors[x]])
    fig.show()
    x+=1


# Time Series Visualization

# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
x=0
for i in prices:
    plot_data = [
        go.Scatter(
            x=df['Month'],
            y=df[i],
            name=i,
            marker = dict(color = colors[x])
            #x_axis="OTI",
            #y_axis="time",
        )
    ]
    plot_layout = go.Layout(
            title=i,
            yaxis_title=i,
            xaxis_title='Month'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)
    x+=1


# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
x=0
for i in percentages:
    plot_data = [
        go.Scatter(
            x=df['Month'],
            y=df[i],
            name=i,
            marker = dict(color = colors[x])
            #x_axis="OTI",
            #y_axis="time",
        )
    ]
    plot_layout = go.Layout(
            title=i,
            yaxis_title=i,
            xaxis_title='Month'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)
    x+=1


# Overlapped Graphs for Comparision

# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
x=0
plot_data=[]
for i in prices:
    item= go.Scatter(
            x=df['Month'],
            y=df[i],
            name=i,
            marker = dict(color = colors[x])
            #x_axis="OTI",
            #y_axis="time",
        )
    plot_data.append(item)
    x+=1
plot_layout = go.Layout(
        title='Overlapped Prices',
        #yaxis_title=i,
        xaxis_title='Month'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
x=0
plot_data=[]
for i in percentages:
    item= go.Scatter(
            x=df['Month'],
            y=df[i],
            name=i,
            marker = dict(color = colors[x])
            #x_axis="OTI",
            #y_axis="time",
        )
    plot_data.append(item)
    x+=1
plot_layout = go.Layout(
        title='Overlapped Price % change',
        #yaxis_title=i,
        xaxis_title='Month'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:




