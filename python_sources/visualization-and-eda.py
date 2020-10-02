#!/usr/bin/env python
# coding: utf-8

# ****EDA****

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import os
os.chdir('../input/Data/Stocks')


# try to read a files of size 600000, because there are too many data  so i choose Top 12 

# In[3]:


files= [x for x in os.listdir() if x.endswith('.txt') and os.path.getsize(x) > 600000]


# In[4]:


files


# In[5]:


mkr = pd.read_csv(files[0],sep=',',index_col='Date')
ibm = pd.read_csv(files[1],sep=',',index_col='Date')
xom = pd.read_csv(files[2],sep=',',index_col='Date')
ba = pd.read_csv(files[3],sep=',',index_col='Date')
dis = pd.read_csv(files[4],sep=',',index_col='Date')
mcd = pd.read_csv(files[5],sep=',',index_col='Date')
utx = pd.read_csv(files[6],sep=',',index_col='Date')
ge = pd.read_csv(files[7],sep=',',index_col='Date')
jnj = pd.read_csv(files[8],sep=',',index_col='Date')
hpq = pd.read_csv(files[9],sep=',',index_col='Date')
ko = pd.read_csv(files[10],sep=',',index_col='Date')
mo = pd.read_csv(files[11],sep=',',index_col='Date')


# In[6]:


mkr.head(10)


# The closing prices are the most interesting.
# A closing table is created. 
# And the data could be empty.
# The seats should be replaced with the highest price on the previous day or the previous day.

# In[7]:


# create close DataFrame
close_price = pd.DataFrame()
close_price['mkr'] = mkr['Close']
close_price['ibm'] = ibm['Close']
close_price['xom'] = xom['Close']
close_price['ba'] = ba['Close']
close_price['dis'] = dis['Close']
close_price['mcd'] = mcd['Close']
close_price['utx'] = utx['Close']
close_price['ge'] = ge['Close']
close_price['jnj'] = jnj['Close']
close_price['ko'] = ko['Close']
close_price['mo'] = mo['Close']


# In[8]:


close_price = close_price.fillna(method='ffill')


# In[9]:


close_price.index =close_price.index.astype('datetime64[ns]')


# In[10]:


close_price.describe()


# In[11]:


close_price.head(10)


# Draw a simple graph of the closing data.

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
_ = pd.concat([close_price['mkr'],close_price['ibm'],close_price['xom'],close_price['ba'],close_price['dis'],close_price['mcd'],close_price['utx'],close_price['ge'],close_price['jnj'],close_price['ko'],close_price['mo']],axis=1).plot(figsize=(20,15),grid=True)


# Because stock shares rise on a 1994 basis when looking at the graph, let's remove the date before 1994 to get a good look at the graph.

# In[13]:


# Remove 1994
from datetime import datetime
close_price = close_price[close_price.index.year>=1994]


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
_ = pd.concat([close_price['mkr'],close_price['ibm'],close_price['xom'],close_price['ba'],close_price['dis'],close_price['mcd'],close_price['utx'],close_price['ge'],close_price['jnj'],close_price['ko'],close_price['mo']],axis=1).plot(figsize=(20,15),grid=True)


# Divide by the maximum value to match the indexes and reschedule.
# Then all maximum value can not but be less than 1.

# In[15]:


company_list = close_price.columns
for idx in company_list:
    close_price[idx+'_scale'] = close_price[idx]/max(close_price[idx])
_ = pd.concat([close_price['mkr_scale'],close_price['ibm_scale'],close_price['xom_scale'],close_price['ba_scale'],close_price['dis_scale'],close_price['mcd_scale'],close_price['utx_scale'],close_price['ge_scale'],close_price['jnj_scale'],close_price['ko_scale'],close_price['mo_scale']],axis=1).plot(figsize=(20,15),grid=True)


# Generally, common elements appear to exist as they rise from 1998 to 2002 and decline around 2009.

# In[16]:


from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix


# In[17]:



fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(15)

for i in company_list:
    if '_scale' not in i:
        _ = autocorrelation_plot(close_price[i],label=i)


# * Most of the companies, except for two, are in decline in 2000, so they seem to have a correlation with each other.
# It shows that an index is about to continue to rise when it is in a rising position, and then to fall when it is a falling market.

# In[18]:


_ = scatter_matrix(pd.concat([close_price['mkr_scale'],close_price['ibm_scale'],close_price['xom_scale'],close_price['ba_scale'],close_price['dis_scale'],close_price['mcd_scale'],close_price['utx_scale'],close_price['ge_scale'],close_price['jnj_scale'],close_price['ko_scale'],close_price['mo_scale']],axis=1),figsize=(20,15),diagonal='kde')


# We find evidence that indexes are strongly correlated, that our premise is working, and that one market can affect another.

# The actual values in some indexes are not useful for modelling. This may be a useful metric, but from a core point of view, time series data is fixed at the mean and therefore there is no trend in data. There are many ways to do this, but all of this is to look at the differences between the values essentially rather than looking at the absolute values. For market data, a common way to deal with recorded results is to calculate the values taken from the natural logs after dividing the indices today by yesterday's index
# 
# There are a number of reasons why log returns are preferred over percentages returns, for example, when a log can be totaled and follows a normal distribution. However, this is not a big deal. All I'm interested in is getting fixed time series data.

# In[19]:


log_data = pd.DataFrame()
for idx in company_list:
    if '_scale' not in idx:
        log_data[idx+'_log'] = np.log(close_price[idx]/close_price[idx].shift())


# In[20]:


log_data.describe()


# In[21]:


_ = pd.concat([log_data['mkr_log'],log_data['ibm_log'],log_data['xom_log'],log_data['ba_log'],log_data['dis_log'],log_data['mcd_log'],log_data['utx_log'],log_data['ge_log'],log_data['jnj_log'],log_data['ko_log'],log_data['mo_log']],axis=1).plot(figsize=(20,15),grid=True)


# In[22]:


fig = plt.figure()
fig.set_figwidth(20)
fig.set_figheight(15)

list_log = log_data.columns

for i in list_log:
    _ = autocorrelation_plot(log_data[i])


# *** Markov process
# An estimation process based on the assumption that the probability of each event among a series of random events depends only on prior results.
# That is, the probability that the future state is determined by the present and that the past is not the variable.

# Nothing comes out of the table of.
# Because financial market is Macos process, knowledge of the past predicts the future.
# It is not helpful. 
# 
# The above data had a centralized and sized mean similar to the time-series data.

# **Let's demonstrate Marcos processes**

# In[23]:


#Let's take a look at the basic correlation.
log_data.corr()


#  So how would that correlate to the data two days ago at this point?

# In[24]:


yesterday = pd.DataFrame()
yesterday['mkr'] = log_data['mkr_log'].shift(1)
yesterday['ibm'] = log_data['ibm_log'].shift(1)
yesterday['xom'] = log_data['xom_log'].shift(1)
yesterday['ba'] = log_data['ba_log'].shift(1)
yesterday['dis'] = log_data['dis_log'].shift(1)
yesterday['mcd'] = log_data['mcd_log'].shift(1)
yesterday['utx'] = log_data['utx_log'].shift(1)
yesterday['ge'] = log_data['ge_log'].shift(1)
yesterday['jnj'] = log_data['jnj_log'].shift(1)
yesterday['ko'] = log_data['ko_log'].shift(1)
yesterday['mo'] = log_data['mo_log'].shift(1)


# In[26]:


all = pd.concat([log_data,yesterday],axis=1)


# In[28]:


all.corr()


# From the results above, the more data you enter into the past, 
# You can see that the correlation is marginal.
# That is, stock forecasting, in line with the Marcos process, is enabled at this time.
# Historical data is not helpful.

# In[ ]:




