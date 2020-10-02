#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
plotly.offline.init_notebook_mode()


# In[ ]:


Meta_data = pd.read_csv('../input/globalterrorismdb_0616dist.csv',encoding='ISO-8859-1',
                        usecols=[0, 1, 2, 3, 8, 11, 13, 14, 29, 35, 84, 100, 103])


# In[ ]:


terror_data = Meta_data.rename(
    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',
             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',
             'weaptype1_txt':'weapon', 'attacktype1_txt':'attack',
             'nkill':'fatalities', 'nwound':'injuries'})


# In[ ]:


terror_IND = terror_data[(terror_data.country == 'India')]


# **Total number of terror attacks in INDIA 1970 - 2015** 

# In[ ]:


len(terror_IND)


# 
# **Terrorist attacks by top 20 Countries**
# 

# In[ ]:


Countries_Terro_Count=Meta_data['country_txt'].value_counts()


# In[ ]:


Countries_Terro_Count.head(20)


# In[ ]:


Countries_Terro_Count.tail(20)


# In[ ]:


INDIA_Terror_Data = terror_data[terror_data['country'].str.contains("India")]


# In[ ]:


len(INDIA_Terror_Data)


# In[ ]:


INDIA_Terror_Data.columns


# In[ ]:


Ind_perstate_Count = pd.DataFrame({'State':INDIA_Terror_Data['state'].value_counts().
                               index, 'Attack Counts':INDIA_Terror_Data['state'].value_counts().values,
                              })


# **State Wise Terrorist Attacks in INDIA***

# In[ ]:


Ind_perstate_Count


# **Terrorist Attack in INDIA**

# In[ ]:


Weapons


# In[ ]:


Weapons=INDIA_Terror_Data['weapon'].value_counts() 


# 
# **Terrorist Attacks by Year in India**

# In[2]:


# terrorist attacks by year
terror_peryear = np.asarray(INDIA_Terror_Data.groupby('year').year.count())

terror_years = np.arange(1990, 2016)

terror_years = np.delete(terror_years, [23])

trace = [go.Scatter(
         x = terror_years,
         y = terror_peryear,
         mode = 'lines+markers',
         name = 'Terror Counts',
         line = dict(
             color = 'Viridis',
             width = 3)
         )]

layout = go.Layout(
         title = 'Terrorist Attacks by Year in INDIA (1990-2016)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.10),
             showline = True,
             showgrid = True
            
         ),
         yaxis = dict(
             range = [0.1, 425],
             showline = True,
             showgrid = True)
         )

figure = dict(data = trace, layout = layout)
iplot(figure)


# In[ ]:




