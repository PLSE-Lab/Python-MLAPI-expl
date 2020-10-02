#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_csv('../input/time_series_covid_19_confirmed.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.drop(columns=['Province/State','Lat','Long'],axis=1,inplace=True)
data


# In[ ]:


# Analyzing repetetions due to difference in Province
n1 = pd.DataFrame(data['Country/Region'].value_counts())
n1[n1['Country/Region'] > 1]


# In[ ]:


n = pd.DataFrame(data['Country/Region'].value_counts())
filt = list(n[n['Country/Region'] > 1].index)
filt


# In[ ]:


ind_lst = list(data[data['Country/Region'].isin(filt)].index)
ind_lst


# In[ ]:


# Test code
test = data[data['Country/Region']=='Australia']
'''for i in range(1,test.shape[1]):
        test.replace(test.iloc[0,i],sum(test.iloc[:,i].values),inplace=True)'''
#test.replace(test.iloc[0,5],sum(test.iloc[:,5].values),inplace=True)
test.iloc[0,5] = sum(test.iloc[:,5].values)
test


# In[ ]:


# Summing up all province based cases to generalize the data for countries only
for cont in filt:
    new = data[data['Country/Region']==cont]
    for i in range(1,new.shape[1]):
        new.iloc[0,i] = sum(new.iloc[:,i].values)
    new.drop_duplicates(subset = ['Country/Region'],keep='first',inplace=True)
    data = data.append(new)


# In[ ]:


data.tail(7)


# In[ ]:


data.drop_duplicates(subset = ['Country/Region'],keep='last',inplace=True)


# In[ ]:


# Checking if our code worked 
data['Country/Region'].value_counts().values > 1


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1750206" data-url="https://flo.uri.sh/visualisation/1750206/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')


# In[ ]:




