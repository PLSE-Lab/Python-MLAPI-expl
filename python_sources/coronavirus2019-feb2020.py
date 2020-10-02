#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/corona.csv")
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


print(df.columns)


# In[ ]:


dff=df.iloc[0:-1,1:]
dff


# In[ ]:


#Percentage of missing values
round((dff.isnull().sum()/len(dff))*100,1)


# In[ ]:


catcols=list(dff.select_dtypes(include=['object']).columns)
print("Categorical columns: ",catcols)


# In[ ]:


numcols=list(dff.select_dtypes(exclude=['object']).columns)
print("Continuous columns: \n",numcols)


# #### Those are null values not missing values, lets impute with 0

# #### Data Imputation

# In[ ]:


dff.fillna(0,inplace=True)
dff.head()


# In[ ]:


dff.info()


# In[ ]:


print("Percentage of missing values: ")
round((dff.isnull().sum()/len(dff))*100,1)


# In[ ]:


dff.rename(columns={"Country,Other":"Country"},inplace=True)
dff.head()


# In[ ]:


dff.describe().T


# In[ ]:


# import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import cufflinks
from plotly.offline import iplot


# In[ ]:


dff.sort_values(by=['TotalCases'],ascending=False,axis=0,inplace=True)
d=dff.head()
d


# In[ ]:


import plotly.graph_objects as go

colors = ['lightslategray',] * 5
colors[0] = 'crimson'

fig = go.Figure()
fig.add_trace(go.Bar(
    x=d["Country"],
    y=d.TotalDeaths,
    
    marker_color=colors,
    text=df.TotalDeaths,
    textposition='auto'
))
fig.update_layout(title_text='Top 5 Countries having Total no. of Deaths')
fig.show()


# In[ ]:


import matplotlib.patches as patches
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5), facecolor='white')
ax.vlines(x=d['Country'], ymin=0, ymax=d['TotalCases'], color='#7B68EE', alpha=0.9, linewidth=20)

# Annotate Text
for i, cty in enumerate(d['TotalCases']):
    ax.text(i, int(cty)+0.1, round(cty, 1), horizontalalignment='center')


# Title, Label, Ticks and Ylim
ax.set_title('Bar Chart for Top 5 Countries having TotalCases', fontdict={'size':15})
ax.set(ylabel='Count')
plt.xticks(d['Country'], d['Country'].str.upper(), rotation=0, horizontalalignment='center', fontsize=10)

# Add patches to color the X axis labels
p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
fig.add_artist(p1)
fig.add_artist(p2)
plt.show()


# ### Top 10 Cities suffering with Corona Virus severely

# In[ ]:


stdeaths=dff.sort_values(by=["TotalDeaths"],axis=0,ascending=False)
d=stdeaths[['Country','TotalDeaths']].values[0:9]
d=pd.DataFrame(d)
d=d.rename({'1':'count'},axis=1)
d.set_index(0,inplace=True)
del d.index.name
d.plot.bar(rot=60)
plt.title('Top 10 Countries suffering with CORONA severely')
print(d)
plt.legend('Deaths')
plt.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=dff["Country"][dff.TotalCases>1000],
    y=dff.TotalCases[dff.TotalCases>1000],
    name='TotalCases',
    marker_color='#FAA460',
    text=df.TotalCases[df.TotalCases>1000],
    textposition='auto'
))
fig.add_trace(go.Bar(
    x=dff["Country"][dff.TotalCases>1000],
    y=dff["TotalRecovered"][dff.TotalCases>1000],
    name='TotalRecovered',
    marker_color='#2ca02c',
    text=dff.TotalRecovered[dff.TotalCases>1000],
    textposition='auto'
))

fig.add_trace(go.Bar(
    x=dff["Country"][dff.TotalCases>1000],
    y=dff.TotalDeaths[dff.TotalCases>1000],
    name='TotalDeaths',
    marker_color='#EF553B',
    text=dff.TotalDeaths[dff.TotalCases>1000],
    textposition='auto'
))
fig.update_layout(title_text='Corona virus TotalCase greater than 1000',xaxis_tickfont_size=14,
    yaxis=dict(
        title='COUNT',
        titlefont_size=16,
        tickfont_size=14))
fig.show()


# ### The countries whose Total Cases are less than 100

# In[ ]:


d=dff[dff['TotalCases']<=100].Country.count()
print("The countries whose Total Cases are less than 100: \n")
print(list(dff[dff['TotalCases']<=100].Country))


# In[ ]:


dff[dff['Country']=='India']


# ### Countries Ranks in their Deaths

# In[ ]:


dff['Rank_Deaths']=dff["TotalDeaths"].rank(ascending=False,method='dense',axis=0)
dff.head()


# In[ ]:


f=dff[dff['Country']=='India'].Rank_Deaths
print("The rank of INDIA in their Death Cases %d out of %d" %(f,dff['Rank_Deaths'].max()))


# ### The countries whose death rate is low

# In[ ]:


print('The countries whose death rate is low: ')
print()
g=dff["Country"][dff["Rank_Deaths"]==17]
print(list(g.unique()))


# ### Gegraphical Representation of all Countries suffering with CORONA VIRUS

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Choropleth(
        locationmode = 'country names',
        locations = dff["Country"],
        z = dff.TotalCases,
        text = dff["Country"],
        colorscale='sunset',
        autocolorscale = False,
        showscale = True,
        geo = 'geo'
    ))
fig.update_layout(title_text='Corona virus affected countries')


# In[ ]:




