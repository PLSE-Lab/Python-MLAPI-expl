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
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


R2015=pd.read_csv('../input/2015MonthwiseRainfallData.csv')
R2016=pd.read_csv('../input/2016MonthwiseRainfallData.csv')
R2017=pd.read_csv('../input/2017MonthwiseRainfallData.csv')


# In[ ]:


R2015.head()


# In[ ]:


Hyd_R2015=R2015[R2015['New_DIST']=='Hyderabad']
Hyd_R2016=R2016[R2016['New_DIST']=='Hyderabad']
Hyd_R2017=R2017[R2017['New_DIST']=='Hyderabad']


# In[ ]:


Hyd_R2015.head()


# In[ ]:


def bar_plot(df,year):
    trace=go.Bar(
        x=df.iloc[:,0],
        y=df.iloc[:,1],
        name=year
    )
    return trace


# In[ ]:


Hyd_R2017['total']=Hyd_R2017['June']+Hyd_R2017['July']+Hyd_R2017['August']+Hyd_R2017['September']


# In[ ]:


fig=go.Figure()


# In[ ]:


fig['layout'].update(title='Total Rainfall in Hyderabad District')
fig.add_trace(bar_plot(Hyd_R2015[['NEW_MANDAL','total']],2015))
fig.add_trace(bar_plot(Hyd_R2016[['NEW_MANDAL','total']],2016))
fig.add_trace(bar_plot(Hyd_R2017[['NEW_MANDAL','total']],2017))


# In[ ]:


R2015.head(2)


# In[ ]:


R2017['total']=R2017['June']+R2017['July']+R2017['August']+R2017['September']


# In[ ]:


all_R2015=R2015.groupby('NEW_MANDAL').mean().reset_index()
all_R2016=R2016.groupby('NEW_MANDAL').mean().reset_index()
all_R2017=R2017.groupby('NEW_MANDAL').mean().reset_index()


# In[ ]:


all_R2015.head()


# In[ ]:


def scatter_plot(df,year):
    trace=go.Scatter(
        x=df.iloc[:,0],
        y=df.iloc[:,1],
        name=year,
        mode='lines+markers'
    )
    return trace


# In[ ]:


subtitles=["Total rainfall in 2015","Total rainfall in 2016","Total rainfall in 2017"]


# In[ ]:


fig2=make_subplots(rows=3,cols=1,subplot_titles=subtitles)


# In[ ]:





# In[ ]:


fig2['layout'].update(title='Total Rainfall District wise',height=1200, width=2000,)
fig2.append_trace((scatter_plot(all_R2015[['NEW_MANDAL','total']],2015)),1,1)
fig2.append_trace((scatter_plot(all_R2016[['NEW_MANDAL','total']],2016)),2,1)
fig2.append_trace((scatter_plot(all_R2017[['NEW_MANDAL','total']],2017)),3,1)


# In[ ]:


fig2.show()


# In[ ]:


all_R2015=R2015.groupby('NEW_MANDAL').mean().reset_index().sort_values(by='total').head()
all_R2016=R2016.groupby('NEW_MANDAL').mean().reset_index().sort_values(by='total').head()
all_R2017=R2017.groupby('NEW_MANDAL').mean().reset_index().sort_values(by='total').head()


# In[ ]:


all_R2015


# In[ ]:


subtitles2=['2015','2016','2017']


# In[ ]:


fig3=make_subplots(rows=1,cols=3,subplot_titles=subtitles2)


# In[ ]:


def Hbar_plot(df,year):
    trace=go.Bar(
        y=df.iloc[:,0],
        x=df.iloc[:,1],
        name=year,
        orientation='h'
    )
    return trace


# In[ ]:





# In[ ]:


fig3['layout'].update(title='Top 5 Districts in Rainfall', width=1500,)
fig3.append_trace((Hbar_plot(all_R2015[['NEW_MANDAL','total']],2015)),1,1)
fig3.append_trace((Hbar_plot(all_R2016[['NEW_MANDAL','total']],2016)),1,2)
fig3.append_trace((Hbar_plot(all_R2017[['NEW_MANDAL','total']],2017)),1,3)


# In[ ]:


fig3.show()


# In[ ]:





# In[ ]:




