#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
get_ipython().run_line_magic('matplotlib', 'inline')


import plotly.tools as tls
import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

print(__version__) # requires version >= 1.9.0
cf.go_offline()


# In[ ]:


df = pd.read_csv('../input/covid-19-geographic-distribution-worldwide/COVID-19-geographic-disbtribution-worldwide.csv')
df.head()


# In[ ]:


df.drop(['DateRep'], axis=1)


# In[ ]:


df_reg=df.groupby(['Countries and territories']).agg({'Cases':'sum','Deaths':'sum'}).sort_values(["Cases"],ascending=False).reset_index()
df_reg.head(10)


# In[ ]:


fig = go.Figure(data=[go.Table(
    columnwidth = [50],
    header=dict(values=('Countries and territories', 'Cases', 'Deaths'),
                fill_color='#104E8B',
                align='center',
                font_size=14,
                font_color='white',
                height=40),
    cells=dict(values=[df_reg['Countries and territories'].head(10), df_reg['Cases'].head(10), df_reg['Deaths'].head(10)],
               fill=dict(color=['#509EEA', '#A4CEF8',]),
               align='right',
               font_size=12,
               height=30))
])

fig.show()


# In[ ]:


df_reg.iplot(kind='box')


# In[ ]:


fig = px.pie(df_reg.head(10),
             values="Cases",
             names="Countries and territories",
             title="Cases",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')
fig.show()


# In[ ]:


fig = px.pie(df_reg.head(10),
             values="Deaths",
             names="Countries and territories",
             title="Deaths",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')
fig.show()


# In[ ]:


df_country=df.groupby(['DateRep','Countries and territories']).agg({'Cases':'sum','Deaths':'sum'}).sort_values(["Cases"],ascending=False)
df_country.head(10)


# In[ ]:


dfd = df_country.groupby('DateRep').sum()
dfd.head()


# In[ ]:


dfd[['Cases','Deaths']].iplot(title = 'World Situation Over Time')


# In[ ]:


dfd['Active'] = dfd['Cases']-dfd['Deaths']
dfd['Active'] 


# In[ ]:


df_country.head()


# In[ ]:


df_glob = df_country.groupby(['DateRep', 'Countries and territories'])['Cases', 'Deaths'].max().reset_index()
df_glob['Dates'] = pd.to_datetime(df_glob['DateRep'])
df_glob['Dates'] = df_glob['Dates'].dt.strftime('%d/%m/%Y')
df_glob['size'] = df_glob['Cases'].pow(0.3)

fig = px.scatter_geo(df_glob, locations='Countries and territories', locationmode='country names', 
                     color="Cases", hover_name="Countries and territories", 
                     range_color= [0, 1500], 
                     projection="mercator", animation_frame='Dates', 
                     title='COVID-19: World Spread', color_continuous_scale="portland")
fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

