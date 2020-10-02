#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
pd.options.mode.chained_assignment = None

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


import plotly.plotly as py


# In[3]:


from plotly import tools


# In[4]:


import plotly.graph_objs as go


# In[5]:


from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()


# In[8]:


data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1',usecols=[0, 1, 2, 3, 8, 11, 13, 14, 35, 82, 98, 101])


# In[11]:


terror_data = data.rename(
    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',
             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',
             'weaptype1_txt':'weapon', 'nkill':'fatalities', 'nwound':'injuries'})


# In[12]:



terror_data['fatalities'] = terror_data['fatalities'].fillna(0).astype(int)
terror_data['injuries'] = terror_data['injuries'].fillna(0).astype(int)


# In[19]:


terror_jordan = terror_data[(terror_data.country == 'Jordan')]# &
terror_jordan['day'][terror_jordan.day == 0] = 1


# In[20]:


terror_jordan['date'] = pd.to_datetime(terror_jordan[['day', 'month', 'year']])


# In[21]:


terror_jordan = terror_jordan[['id', 'date', 'year', 'state', 'latitude', 'longitude',
                         'target', 'weapon', 'fatalities', 'injuries']]


# In[22]:


terror_jordan = terror_jordan.sort_values(['fatalities', 'injuries'], ascending = False)


# In[23]:


terror_jordan = terror_jordan.drop_duplicates(['date', 'latitude', 'longitude', 'fatalities'])


# In[24]:


terror_jordan['text'] = terror_jordan['date'].dt.strftime('%B %-d, %Y') + '<br>' +                     terror_jordan['fatalities'].astype(str) + ' Killed, ' +                     terror_jordan['injuries'].astype(str) + ' Injured'


# In[27]:


fatality = dict(
           type = 'scattergeo',
           locationmode = 'Jordan',
           lon = terror_jordan[terror_jordan.fatalities > 0]['longitude'],
           lat = terror_jordan[terror_jordan.fatalities > 0]['latitude'],
           text = terror_jordan[terror_jordan.fatalities > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_jordan[terror_jordan.fatalities > 0]['fatalities'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )


# In[29]:


injury = dict(
         type = 'scattergeo',
         locationmode = 'Jordan',
         lon = terror_jordan[terror_jordan.fatalities == 0]['longitude'],
         lat = terror_jordan[terror_jordan.fatalities == 0]['latitude'],
         text = terror_jordan[terror_jordan.fatalities == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_jordan[terror_jordan.fatalities == 0]['injuries'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )


# In[30]:


layout = dict(
         title = 'Terrorist Attacks by Latitude/Longitude in Jordan (1970-2015)',
         showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
         geo = dict(
             scope = 'Jordan',
             projection = dict(type = 'albers jordan'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )


# In[31]:


data = [fatality, injury]


# In[32]:


figure = dict(data = data, layout = layout)


# In[ ]:


iplot(figure)


# In[ ]:




