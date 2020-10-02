#!/usr/bin/env python
# coding: utf-8

# The word is climate change is one of the biggest existential threat that humanity is facing. Hoping to throw some exploratory light on the matter with the given data. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from matplotlib import pyplot as plt
import seaborn as sbn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


global_temperatures = pd.read_csv("../input/GlobalTemperatures.csv", infer_datetime_format=True, index_col='dt', parse_dates=['dt'])
print (global_temperatures.info())


# In[ ]:


global_temperatures[global_temperatures.index.year > 2000]['LandAverageTemperature'].plot(figsize=(13,7))


# The oscillation basically depicts the seasonal variance in average temperature - To gain a better insight let's try grouping the average temperature by year and plotting the average temperature change over years.

# In[ ]:


global_temperatures.groupby(global_temperatures.index.year)['LandAverageTemperature'].mean().plot(figsize=(13,7))


# That seems about correct. I'm guessing the instruments we had in the early years had huge uncertainty, which is why we see the data in the initial years with large variation - as seen in below plot. 
# Anyways bottom line is the average temperature has gradually increased over the year as seen in the plot.
# 
# Note: pandas rolling mean function with window = 12 will not provide the analysis we are looking for.

# In[ ]:


global_temperatures.groupby(global_temperatures.index.year)['LandAverageTemperatureUncertainty'].mean().plot(figsize=(13,7))

