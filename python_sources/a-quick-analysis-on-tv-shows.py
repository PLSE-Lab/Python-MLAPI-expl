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


df = pd.read_csv('/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')


# In[ ]:


df


# In[ ]:


df['Rotten_tomatoes_%']=df['Rotten Tomatoes'].str.replace('%', '')


# In[ ]:


df['Rotten_tomatoes_%']=df['Rotten_tomatoes_%'].astype('float')


# In[ ]:


df.describe()


# In[ ]:


#Correlation matrix
corrmat = df.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True, annot = True)
plt.show()


# In[ ]:


topimdb=df.sort_values(by=['IMDb'], ascending=False)


# In[ ]:


top10imdb=topimdb.head(10)


# In[ ]:


top10imdb


# In[ ]:


import plotly.express as px
fig = px.bar(top10imdb, y="IMDb", x="Title", color='IMDb')
fig.show()


# In[ ]:


yr = df['Year'].value_counts()
yr


# In[ ]:


yr=pd.DataFrame(yr)


# In[ ]:


yr=yr.reset_index()


# In[ ]:


yr=yr.rename(columns={"Year": "Count"})


# In[ ]:


yr=yr.rename(columns={"index": "Year"})


# In[ ]:


import plotly.express as px
fig = px.bar(yr, x="Year", y="Count", color='Count')
fig.show()


# In[ ]:


age = df['Age'].value_counts()
age


# In[ ]:


age=age.reset_index()


# In[ ]:


age=age.rename(columns={"Age": "Count"})


# In[ ]:


age=age.rename(columns={"index": "Age"})


# In[ ]:


import plotly.express as px
fig = px.bar(age, x="Age", y="Count", color='Count')
fig.show()


# In[ ]:


IMDb = df['IMDb'].value_counts()
IMDb


# In[ ]:


IMDb=pd.DataFrame(IMDb)


# In[ ]:


IMDb=IMDb.reset_index()


# In[ ]:


IMDb=IMDb.rename(columns={"IMDb": "Count"})


# In[ ]:


IMDb=IMDb.rename(columns={"index": "IMDb"})


# In[ ]:


import plotly.express as px
fig = px.bar(IMDb, x="IMDb", y="Count", color='Count')
fig.show()


# In[ ]:


import plotly.express as px
#df = px.data.tips()
fig = px.histogram(df, x="Rotten_tomatoes_%", nbins=100, opacity=0.8,
                   color_discrete_sequence=['indianred'])
fig.show()


# In[ ]:


netflix=df[df['Netflix']==1]
hulu=df[df['Hulu']==1]
prime=df[df['Prime Video']==1]
disney=df[df['Disney+']==1]
channels=[netflix, hulu, prime, disney]
cols=['Year', 'Age', 'IMDb']


# In[ ]:


dflist=[]
k=0


# In[ ]:


for i in channels:
    collist=[]
    for j in cols:
        a = i[j].value_counts()
        a=pd.DataFrame(a)
        a=a.reset_index()
        a=a.rename(columns={j: "Count"})
        a=a.rename(columns={"index": j})
        collist.append(a)
    dflist.append(collist)
    


# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Bar(x=dflist[0][0]["Year"], y=dflist[0][0]["Count"], name='Netflix'),
    go.Bar(x=dflist[1][0]["Year"], y=dflist[1][0]["Count"], name='Hulu'),
    go.Bar(x=dflist[2][0]["Year"], y=dflist[2][0]["Count"], name='Prime Video'),
    go.Bar(x=dflist[3][0]["Year"], y=dflist[3][0]["Count"], name='Disney+')
]
plot_layout = go.Layout(
        title='Year',
        yaxis_title='Count',
        xaxis_title='Year'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Bar(x=dflist[0][1]["Age"], y=dflist[0][1]["Count"], name='Netflix'),
    go.Bar(x=dflist[1][1]["Age"], y=dflist[1][1]["Count"], name='Hulu'),
    go.Bar(x=dflist[2][1]["Age"], y=dflist[2][1]["Count"], name='Prime Video'),
    go.Bar(x=dflist[3][1]["Age"], y=dflist[3][1]["Count"], name='Disney+')
]
plot_layout = go.Layout(
        title='Age',
        yaxis_title='Count',
        xaxis_title='Age'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Bar(x=dflist[0][2]["IMDb"], y=dflist[0][2]["Count"], name='Netflix'),
    go.Bar(x=dflist[1][2]["IMDb"], y=dflist[1][2]["Count"], name='Hulu'),
    go.Bar(x=dflist[2][2]["IMDb"], y=dflist[2][2]["Count"], name='Prime Video'),
    go.Bar(x=dflist[3][2]["IMDb"], y=dflist[3][2]["Count"], name='Disney+')
]
plot_layout = go.Layout(
        title='IMDb',
        yaxis_title='Count',
        xaxis_title='IMDb'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:


import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Histogram(x=netflix["Rotten_tomatoes_%"], name='Netflix'),
    go.Histogram(x=hulu["Rotten_tomatoes_%"], name='Hulu'),
    go.Histogram(x=prime["Rotten_tomatoes_%"], name='Prime Video'),
    go.Histogram(x=disney["Rotten_tomatoes_%"], name='Disney+')
]
plot_layout = go.Layout(
        title='Rotten Tomatoes',
        yaxis_title='Count',
        xaxis_title='Rotten Tomatoes %'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:




