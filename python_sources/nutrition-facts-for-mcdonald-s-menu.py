#!/usr/bin/env python
# coding: utf-8

# by xiaoyao70v

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


menu = pd.read_csv('../input/menu.csv')
menu.head()


# In[ ]:


print(menu.info())


# In[ ]:


f,axes = plt.subplots(3,3,figsize=(10,10),sharex = True,sharey =True)
s=np.linspace(0,3,10)
cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)

x = menu['Cholesterol (% Daily Value)'].values
y = menu['Sodium (% Daily Value)'].values
sns.kdeplot(x,y,cmap=cmap,shade=True,cut = 5,ax=axes[0,0])
axes[0,0].set(xlim=(-10,50),ylim=(-30,70),title="Cholesterol and Sodium")

cmap = sns.cubehelix_palette(start =0.333333333333,light = 1 ,as_cmap = True )
x = menu['Carbohydrates (% Daily Value)'].values
y = menu['Sodium (% Daily Value)'].values
sns.kdeplot(x,y,cmap=cmap,shade=True,cut=5,ax=axes[0,1])
axes[0,1].set(xlim=(-5,50),ylim=(-10,70),title='Carbs and Sodium')

cmap =sns.cubehelix_palette(start = 0.666666666667,light = 1,as_cmap = True)
x = menu['Carbohydrates (% Daily Value)'].values
y = menu['Cholesterol (% Daily Value)'].values
sns.kdeplot(x,y,cmap=cmap,shede=True,cut=5,ax=axes[0,2])
axes[0,2].set(xlim=(-5,50),ylim=(-10,70),title='Carbs and Cholesterol')

cmap = sns.cubehelix_palette(start = 1.0 ,light = 1,as_cmap=True)
x =menu['Total Fat (% Daily Value)'].values
y = menu['Saturated Fat (% Daily Value)'].values
sns.kdeplot(x,y,cmap=cmap,shade=True,ax=axes[1,0])
axes[1,0].set(xlim=(-5,50),ylim=(-10,70),title='Total Fat and Saturated Fat')

cmap =sns.cubehelix_palette(start = 1.333333333,light = 1,as_cmap=True)
x = menu['Cholesterol (% Daily Value)'].values
y = menu['Total Fat (% Daily Value)'].values
sns.kdeplot(x,y,cmap=cmap,shade=True,cut=5,ax=axes[1,1])
axes[1,1].set(xlim=(-5,50),ylim=(-10,70),title='Cholesterol and Total Fat')

cmap =sns.cubehelix_palette(start = 1.66666667 ,light=1, as_cmap = True)
x = menu['Vitamin A (% Daily Value)'].values
y = menu['Cholesterol (% Daily Value)'].values
sns.kdeplot(x,y,shade=True,cmap=cmap,cut=5,ax=axes[1,2])
axes[1,2].set(xlim=(-5,50),ylim=(-10,70),title='Vitamin A and Cholesterol')

cmap =sns.cubehelix_palette(start = 2.0,light = 1,as_cmap = True)
x = menu['Calcium (% Daily Value)'].values
y = menu['Sodium (% Daily Value)'].values
sns.kdeplot(x,y,cmap=cmap,shade = True,ax=axes[2,0])
axes[2,0].set(xlim=(-5,50),ylim=(-10,70),title='Calcium and Sodium')

cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)

# Generate and plot a random bivariate dataset
x = menu['Calcium (% Daily Value)'].values
y = menu['Cholesterol (% Daily Value)'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[2,1])
axes[2,1].set(xlim=(-5, 50), ylim=(-10, 70),  title = 'Cholesterol and Calcium')

cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)

# Generate and plot a random bivariate dataset
x = menu['Iron (% Daily Value)'].values
y = menu['Total Fat (% Daily Value)'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[2,2])
axes[2,2].set(xlim=(-5, 50), ylim=(-10, 70),  title = 'Iron and Total Fat')


f.tight_layout()


# In[ ]:


trace = go.Scatter(
    y = menu['Cholesterol (% Daily Value)'].values,
    x = menu['Item'].values,
    mode = 'markers',
    marker = dict(
        size = menu['Cholesterol (% Daily Value)'].values,
        color = menu['Cholesterol (% Daily Value)'].values,
        colorscale = 'Protland',
        showscale = True),
    text = menu['Item'].values)
data = [trace]

layout = go.Layout(
                    autosize = True,
                    title = 'Scatter plot of Cholesterol (% Daily Value) per Item on Menu',
                    hovermode = 'closest',
                    xaxis=dict(
                                showgrid = False,
                                zeroline = False,
                                showline = False),
                    yaxis = dict(
                    title = 'Cholesterol (% Daily Value)',
                    ticklen = 5,
                    gridwidth = 2,
                    showgrid =False,
                    zeroline = False,
                    showline = False),
                    showlegend = False)

fig = go.Figure(data = data,layout = layout)
py.iplot(fig,filename = 'scatterChol')


# In[ ]:




