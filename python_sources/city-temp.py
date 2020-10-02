#!/usr/bin/env python
# coding: utf-8

#     This is my second analysis with real data. I hope you like it. The variation of 2 for temperature is an important fact, since we expect this variation in 100 years and not for 24 years.
#      I could not fine a standard for the 5 countries I analyzed. The std contains some high values wich might be produced by the variety of locations that temperature may have been measured.

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


data=pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')


# In[ ]:


data.drop(columns='State', inplace=True)


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
pd.set_option('display.max_rows', None)
from plotly.subplots import make_subplots
import seaborn as sns
import datetime


# In[ ]:


data.shape


# In[ ]:


data['Year']=data['Year'].astype('str')


# time data **201**-12-3 doesn't match format specified
# 

# time data 2008-3-**0** doesn't match format specified

# In[ ]:


# 8 rows with day=0
data=data.loc[data.Day!=0]


# In[ ]:


data.shape


# In[ ]:


def counting(year):
    c= len(year)
    if c<4:
        return 0
    else:
        return year


# In[ ]:


data['Year']=data.Year.apply(counting)


# In[ ]:


#440 rows with year= 202
data = data.loc[data.Year!=0]


# In[ ]:


data.shape


# ## Mean of temperature by city

# In[ ]:


data.head()


# In[ ]:


data=data.loc[data.AvgTemperature!=-99]


# In[ ]:


data.head()


# In[ ]:


data['Year']=data.Year.astype('int64')


# In[ ]:


data_1995=data.loc[data.Year==1995]


# In[ ]:


data_1995.head()


# In[ ]:


year1995=data_1995.groupby(['Year','Region','Country']).agg(Mean_temp=('AvgTemperature','mean'),Mean_std=('AvgTemperature','std'))


# In[ ]:


year1995.reset_index(inplace=True)


# In[ ]:


year1995.head()


# In[ ]:


fig = px.choropleth(year1995, locations=year1995['Country'],
                    color=year1995['Mean_temp'],locationmode='country names', 
                    hover_name=year1995['Country'], 
                    color_continuous_scale=px.colors.sequential.Tealgrn,template='plotly_dark', )
fig.update_layout(
    title='Temperature in 1995',
)
fig.show()


# In[ ]:


year2019=data.loc[data.Year==2019]


# In[ ]:


year2019=year2019.groupby(['Year','Region','Country']).agg(Mean_temp=('AvgTemperature','mean'),Mean_std=('AvgTemperature','std'))


# In[ ]:


year2019.reset_index(inplace=True)


# In[ ]:


year2019.head()


# In[ ]:


fig = px.choropleth(year2019, locations=year2019['Country'],
                   color=year2019['Mean_temp'], locationmode='country names',
                   hover_name=year2019['Country'],
                   color_continuous_scale=px.colors.sequential.Tealgrn,template='plotly_dark')
fig.update_layout(
    title='Temperature in 2019',
)
fig.show()


# In[ ]:


data.head()


# ## Graphic for the top 5 most polluting countries (China, EUA, India, Russia, Japan)

# In[ ]:


polu=data.loc[(data.Country=='China') | (data.Country=='US') | (data.Country=='India') | (data.Country=='Russia') | (data.Country=='Japan') &(data.Year!=2020)]


# In[ ]:


polu=polu.loc[polu.Year!=2020]


# In[ ]:


polu2=polu.groupby(['Year','Country']).agg(Mean_temp=('AvgTemperature','mean'),Mean_std=('AvgTemperature','std'))


# In[ ]:


polu2.head(10)


# In[ ]:


polu2.reset_index(inplace=True)


# In[ ]:


polu2.head(10)


# In[ ]:


fig=px.bar(polu2, x='Country', y='Mean_temp', error_y='Mean_std',animation_frame='Year', animation_group='Country', hover_name='Country', range_y=[0,100])
fig.show()


# ### Analyzing the variation for the same month between the years of these countries month = January (1)

# In[ ]:


polu_1=polu.loc[polu.Month==1]


# In[ ]:


polu_m=polu_1.groupby(['Year','Country','Month']).agg(Mean_temp=('AvgTemperature','mean'), Std_temp=('AvgTemperature','std'))


# In[ ]:


polu_m.reset_index(inplace=True)


# In[ ]:


polu_m.head()


# In[ ]:


fig=px.bar(polu_m, x='Country', y='Mean_temp', error_y='Std_temp', animation_frame='Year', animation_group='Country', hover_name='Country', range_y=[0,90])
fig.show()


# In[ ]:




