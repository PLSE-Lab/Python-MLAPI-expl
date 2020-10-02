#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


globalTempdf = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv", parse_dates = True, infer_datetime_format = True)


# In[ ]:


globalTempdf['dt'] = pd.to_datetime(globalTempdf.dt)


# In[ ]:


ZurichTempdf = globalTempdf[globalTempdf.City == 'Zurich'].dropna()
ZurichTempdf.set_index(ZurichTempdf.dt, inplace = True)
ZurichTempdf.tail()


# In[ ]:


zurByYeardf = ZurichTempdf['1/1/1900':'8/1/2013'].resample('A').dropna()
g = sns.tsplot(zurByYeardf.AverageTemperature)


# In[ ]:


plot = plt.plot(zurByYeardf.index, pd.stats.moments.ewma(zurByYeardf.AverageTemperature, com = 14.5))


# In[ ]:


plot1 = plt.plot(zurByYeardf.index, pd.stats.moments.ewma(zurByYeardf.AverageTemperatureUncertainty, com = 9.5))


# Ran out of time.
# 
# To be continued...

# In[ ]:





# In[ ]:





# In[ ]:




