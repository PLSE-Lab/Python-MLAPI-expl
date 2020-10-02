#!/usr/bin/env python
# coding: utf-8

# ## Suicide is a major concern across the world. And people who tell, that suicides are committed by weak people they are pretty much wrong. Only the person who is committing suicide knows what he or she is going through.
# 
# ### This kernel mainly focuses on the analysis of the no. of suicides across different nations, and maybe how we can deal with them. We should be coming up with some tactic to deal with mental patients, because its a very serious issue.
# -------------------------------------------------------------------------------------------------------------------
# 
# ###   Importing the libraries and the dataset

# In[ ]:


import os
print(os.listdir("../input"))

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import warnings
warnings.filterwarnings('ignore')
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm


# In[ ]:


df=pd.read_csv('../input/who_suicide_statistics.csv')
df.head()


#  ### Explaratory Data Analysis (EDA)

# #### What are the different nations where suicides are happening?

# In[ ]:


print("The data has been given for:",df['country'].nunique(),'different countries across the world')


# In[ ]:


print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())


# #### Let's find out which country and which year did maximum no. of suicides happen

# In[ ]:


df.fillna(df.mean(),inplace=True)
df_sui=pd.DataFrame(df.groupby(['country','year'])['suicides_no'].sum().reset_index())
df_sui.head()


# In[ ]:


count_max_sui=pd.DataFrame(df_sui.groupby('country')['suicides_no'].sum().reset_index())

count = [ dict(
        type = 'choropleth',
        locations = count_max_sui['country'],
        locationmode='country names',
        z = count_max_sui['suicides_no'],
        text = count_max_sui['country'],
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick =False,
            title = 'Suicides Country-based'),
      ) ]
layout = dict(
    title = 'Suicides happening across the Globe',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=count, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )


# We see that Russian Federation is the country where max. no. of suicides have happened.

# ### Top 10 ountries having the max. suicides over the years

# In[ ]:


df_sui_n=pd.DataFrame(df.groupby(['country'])['suicides_no'].sum().reset_index())
df_sui_n.sort_values(by=['suicides_no'],ascending=False,inplace=True)


# In[ ]:


type_colors = ['#5D6D7E',  # Grass
                    '#633974',  # Fire
                    '#E74C3C',  # Water
                    '#283747',  # Bug
                    '#F4D03F',  # Normal
                    '#0B5345',  # Poison
                    '#154360',  # Electric
                    '#7B7D7D',  # Ground
                    '#229954',  # Fairy
                    '#641E16',  # Fighting
                   ]


fig, ax = plt.subplots()

fig.set_size_inches(13.7, 10.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df_sui_n["country"].head(10), y=df_sui_n['suicides_no'].head(10), data=df_sui_n,palette=type_colors)
f.set_xlabel("Name of Country",fontsize=18)
f.set_ylabel("No. of Suicides",fontsize=18)
f.set_title('Top 10 Countries having highest no. suicides')
for item in f.get_xticklabels():
    item.set_rotation(90)


# I don't know why, one of the most important country, India's data is missing. India is one of the countries having highest no. of suicides almost every year. 

# In[ ]:


fig = {
  "data": [
    {
      "values": df['sex'].value_counts(),
      "labels": df['sex'].unique(),
        'marker': {'colors': ['rgb(58, 21, 56)',
                                  'rgb(33, 180, 150)']},
      "name": "Gender based suicides",
      "hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    }],
     "layout": {
        "title":"Males or Females, who are more prone to committing suicides?"
     }
}
iplot(fig, filename='donut')


# #### Finding out how the no. of suicides amongst males and females across the world has changed over the years.

# In[ ]:


df_sui_ny=pd.DataFrame(df.groupby(['year','sex'])['suicides_no'].sum().reset_index())
df_sui_ny=df_sui_ny.round(0)
df_sui_ny.sort_values(by=['year'],inplace=True)
df_sui_ny=df_sui_ny.ix[df_sui_ny['year']!=2016] #data seems to missing for this year
df_sui_nym=df_sui_ny.ix[df_sui_ny['sex']=='male']
df_sui_nyf=df_sui_ny.ix[df_sui_ny['sex']=='female']
trace_high = go.Scatter(
                x=df_sui_nym.year,
                y=df_sui_nym['suicides_no'],
                name = "Suicides amongst Males",
                line = dict(color = '#ADAF06'),
                opacity = 0.8)

trace_low = go.Scatter(
                x=df_sui_nyf.year,
                y=df_sui_nyf['suicides_no'],
                name = "Suicides amongst Females",
                line = dict(color = '#870379'),
                opacity = 0.8)

data = [trace_high,trace_low]

layout = dict(
    title = "Suicides amongst Males & Females over the years ",
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Suicides")


# ### Now we'll see how the suicides have happened amongst the genders in the top 10 countries 

# In[ ]:


df_gen=pd.DataFrame(df.groupby(['country','sex'])['suicides_no'].sum()).reset_index()
df_gen=pd.merge(df_gen,pd.DataFrame(df_gen.groupby(['country'])['suicides_no'].sum()).reset_index(),on=['country'])
df_gen.rename(columns={'suicides_no_x':'gender_suicides','suicides_no_y':'total_suicides'},inplace=True)
df_gen.sort_values(by=['total_suicides'],ascending=False,inplace=True)
df_gen.head()


# In[ ]:


df_gen_m=df_gen.ix[df_gen['sex']=="male"]
df_gen_fm=df_gen.ix[df_gen['sex']=="female"]
trace1 = go.Bar(
    x=df_gen_m['country'].head(10),
    y=df_gen_m['gender_suicides'].head(10),
    name='Male Suicides'
)
trace2 = go.Bar(
    x=df_gen_fm['country'].head(10),
    y=df_gen_fm['gender_suicides'].head(10),
    name='Female Suicides'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title='Suicide Distribution amongst the Genders in top 10 countries having max. suicides'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')


# Since Russian Federation is the country having highest no. of suicides, we will concentrate on that country for the time-being. 

# In[ ]:


df_sui_ny=pd.DataFrame(df.groupby(['country','sex','year'])['suicides_no'].sum().reset_index())
df_sui_ny=df_sui_ny.ix[df_sui_ny['country']=="Russian Federation"]
df_sui_ny=df_sui_ny.round(0)
df_sui_ny.sort_values(by=['year'],inplace=True)
df_sui_ny=df_sui_ny.ix[df_sui_ny['year']!=2016] #data seems to missing for this year
df_sui_nym=df_sui_ny.ix[df_sui_ny['sex']=='male']
df_sui_nyf=df_sui_ny.ix[df_sui_ny['sex']=='female']
trace_high = go.Scatter(
                x=df_sui_nym.year,
                y=df_sui_nym['suicides_no'],
                name = "Suicides amongst Males",
                line = dict(color = '#0A0993'),
                opacity = 0.8)

trace_low = go.Scatter(
                x=df_sui_nyf.year,
                y=df_sui_nyf['suicides_no'],
                name = "Suicides amongst Females",
                line = dict(color = '#04B77A'),
                opacity = 0.8)

data = [trace_high,trace_low]

layout = dict(
    title = "Suicides amongst Males & Females over the years in Russian Federation",
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Suicides")


# Its good to see that it has been decreasing over the years.
# 
# ### Next we go on to the time series prediction for the country and then we will do for the whole world.

# How has the no. of suicides, totally varied in the last 30-35 years

# In[ ]:


df_sui_ny=pd.DataFrame(df_sui_ny.groupby(['year','country'])['suicides_no'].sum()).reset_index()

df_ts=df_sui_ny[['year','suicides_no']]
df_ts.index=df_ts['year']


df_ts['suicides_no'].plot(figsize=(15,6), color="red")
plt.xlabel('Year')
plt.ylabel('Total no. of suicides')
plt.title("No. of suicides that happened in last 30-35 years in Russian federation")
plt.show()


# We see that there has been a sharp dip around 1983-85, maybe the data is corrupted for those two years, however we see that the suicides are going exponentially down now.

# In[ ]:


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y')
data=pd.read_csv('../input/who_suicide_statistics.csv', parse_dates=['year'], index_col='year',date_parser=dateparse)
data.fillna(data.mean(),inplace=True)
data=data.reset_index()
data=pd.DataFrame(data.groupby(['country','year'])['suicides_no'].sum()).reset_index()
data=data.sort_values(by=['suicides_no'],ascending=False)
data.head()


# We'll first try to the forecasting for Russian Federation, then we'll look forward to the whole world

# In[ ]:


data=data.ix[data['country']=='Russian Federation']
data.drop('country',axis=1,inplace=True)
data = data.set_index('year')
data.head()


# In[ ]:


from fbprophet import Prophet

df_prophet = data.copy()
df_prophet.reset_index(drop=False,inplace=True)
df_prophet.columns = ['ds','y']
df_prophet=df_prophet[21:]

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=5,freq='Y')
forecast = m.predict(future)
fig = m.plot(forecast)


# In[ ]:


m.plot_components(forecast);


# ### Here is the time-series forecasting till 2020 for Russian Federation

# In[ ]:




