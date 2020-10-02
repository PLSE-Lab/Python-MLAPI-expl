#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size=\"6\">Earthquakes animation using bubbly</font></center></h1>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>
# - <a href='#2'>Earthquakes data</a>
# - <a href='#3'>Timeline using bubbly</a>
# - <a href='#4'>References</a>

# # <a id="1">Introduction</a>
# 
# Significant Earthquakes data <a href='#4'>[1]</a> contains date, time and location of all earthquakes with magnitude of 5.5 or higher for the years 1965-2016.  
# 
# We will show this data in an animation using bubbly <a href='#4'>[2]</a> <a href='#4'>[3]</a>, a python package that extends plotly <a href='#4'>[4]</a>.

# # <a id="1">Earthquakes data</a>
# 
# ## Load the packages

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from bubbly.bubbly import bubbleplot 
from __future__ import division
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
IS_LOCAL = False
import os
if(IS_LOCAL):
    PATH="../input/earthquake-database"
else:
    PATH="../input"
print(os.listdir(PATH))


# ## Load and check the data

# In[ ]:


events_df = pd.read_csv(PATH+"/database.csv")


# In[ ]:


print("Earthaquakes data -  rows:",events_df.shape[0]," columns:", events_df.shape[1])


# In[ ]:


events_df.head(5)


# There are multiple types of events, let's check the types:

# In[ ]:


events_df['Type'].unique()


# We will select only the **Earthquake** types for the timeline.  

# 

# # <a id="3">Timeline using bubbly</a>
# 
# Let's prepare the data. We select only the **Earthquake** type of events. 
# 
# 
# 
# 

# In[ ]:


earthquakes = events_df[events_df['Type']=='Earthquake'].copy()


# We then extract from the **Date** the **Year**.  The year will be used in the timeline, for each year one animation frame being displayed.

# In[ ]:


earthquakes["Year"] = pd.to_datetime(earthquakes['Date']).dt.year


# Let's confirm the limits (in terms of years and magnitude) of the values our earthquakes dataset. 

# In[ ]:


print("Years from:", min(earthquakes['Year']), " to:", max(earthquakes['Year']))
print("Magnitude from:", min(earthquakes['Magnitude']), " to:", max(earthquakes['Magnitude']))


# The Magnitude scale is logarithmic, i.e. **each whole number increase in magnitude represents a tenfold increase in measured amplitude** <a href='#4'>[5]</a>. To represent correctly the relative amplitude of earthquakes of different magnitude we create also a variable (**RichterM**), calculated as following:

# In[ ]:


earthquakes["RichterM"] = np.power(earthquakes["Magnitude"],10)


# The bubble plot (inspired by the work of Rosling <a href='#4'>[6]</a>) is showing the data in (Longitude, Latitude) coordinates, animated, each animation frame showing one year. Each bubble shows a major earthquake.  The size of the bubbles are correlated with the amplitude (calculated from the magnitude). The bubble displays the magnitude (on Richter scale).

# In[ ]:


figure = bubbleplot(dataset=earthquakes, x_column='Longitude', y_column='Latitude', color_column = 'Magnitude',
    bubble_column = 'Magnitude', time_column='Year', size_column = 'RichterM',
    x_title='Longitude', y_title='Latitude', 
    title='Earthquakes position (long, lat) and magnitude - from 1965 to 2016', 
    colorscale='Rainbow', colorbar_title='Magnitude', 
    x_range=[-181,181], y_range=[-90,90], scale_bubble=0.5, height=650)
iplot(figure, config={'scrollzoom': True})


# # <a id="4">References</a>  
# 
# 
# [1] Significant Earthquakes, 1965-2016, Date, time, and location of all earthquakes with magnitude of 5.5 or higher, https://www.kaggle.com/usgs/earthquake-database    
# [2] Aashita Kesarwani, https://www.kaggle.com/aashita/guide-to-animated-bubble-charts-using-plotly  
# [3] Aashita Kesarwani, https://github.com/AashitaK/bubbly/blob/master/bubbly/bubbly.py  
# [4] Plotly, https://community.plot.ly/    
# [5] Richter magnitude scale, Wikipedia, https://en.wikipedia.org/wiki/Richter_magnitude_scale    
# [6] Hans Rosling, https://en.wikipedia.org/wiki/Hans_Rosling  
