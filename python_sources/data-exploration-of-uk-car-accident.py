#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will explore the car accident data and figure out the key factor of accidents.

# In[ ]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="white")


# In[ ]:


with open('../input/Accidents0515.csv', 'r') as f:
    acc = pd.read_csv(f,encoding='utf-8')


# In[ ]:


acc.head()


# In[ ]:


plt.scatter(acc.Longitude,acc.Latitude,c = acc.Police_Force)


# In[ ]:


plt.scatter(acc.Longitude,acc.Latitude,c = acc.Accident_Severity)


# In[ ]:


acc_count = acc.groupby(acc.Accident_Severity).Accident_Severity.count().plot(kind = 'bar')


# In[ ]:


acc_count = acc.groupby(acc.Light_Conditions).Accident_Severity.count().plot(kind = 'bar')


# In[ ]:


acc_count = acc.groupby(acc.Road_Surface_Conditions).Accident_Severity.count().plot(kind = 'bar')


# In[ ]:


acc_count = acc.groupby(acc.Weather_Conditions).Accident_Severity.count().plot(kind = 'bar')


# In[ ]:


acc.groupby(acc.Number_of_Vehicles).Accident_Severity.count().plot(kind = 'bar')


# In[ ]:


#most accident caused by 1,2,3 cars


# In[ ]:


corrmat = acc.corr()


# In[ ]:


f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat,vmax =.8,square = True);


# In[ ]:




