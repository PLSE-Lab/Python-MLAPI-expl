#!/usr/bin/env python
# coding: utf-8

# ###This notebook compares number of attacks during bush(2004-2008) and obama(2009-2017) administration.###

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[ ]:


plt.style.use('ggplot')


# In[ ]:


pd.set_option('display.max_colwidth', 0, 'display.max_columns', 0)


# In[ ]:


data = pd.read_csv('../input/PakistanDroneAttacksWithTemp.csv', encoding='latin1')


# In[ ]:


data.head(2)


# In[ ]:


data['Month'] = pd.DatetimeIndex(pd.to_datetime(data.Date)).month.astype(np.int32)
data['Year'] = pd.DatetimeIndex(pd.to_datetime(data.Date)).year.astype(np.int32)
data['DayOfWeek'] = pd.DatetimeIndex(pd.to_datetime(data.Date)).dayofweek.astype(np.int32)
data.drop(['Date'], axis=1, inplace=True)
data.drop(data[data.Year == -2147483648].index, inplace=True)


# In[ ]:


data.head(2)


# In[ ]:


by_year = data.groupby('Year').size()
ob, bu = {}, {}
for i, j in zip(by_year.keys(), by_year.values):
    if (int(i) >= 2009):
        ob[i] = j
    else:
        bu[i] = j


# In[ ]:


list(bu.keys())


# In[ ]:


fig = plt.figure(figsize=(10, 5))
a = plt.gca()

#plt.plot(list(bu.values()), label='Bush Administration', marker='o')
#plt.plot(list(ob.values()), label='Obama Administration', marker='o')

plt.bar(left=list(bu.keys()), height=list(bu.values()), color=cm.Reds(200), label='Bush Administration')
plt.bar(left=list(ob.keys()), height=list(ob.values()), color=cm.Blues(500), label='Obama Administration')

plt.xlabel('Year')
plt.ylabel('Drone Attacks')
plt.title('Bush VS. Obama Administration')
plt.xlim(2004, 2017)
plt.legend()
a.xaxis.grid(False)


# In[ ]:




