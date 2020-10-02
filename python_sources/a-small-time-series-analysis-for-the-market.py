#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model as lm
import kagglegym

get_ipython().run_line_magic('matplotlib', 'inline')
# Create environment
env = kagglegym.make()

# Get first observation
observation = env.reset()

# Get the train dataframe
train = observation.train


# In[ ]:


df = train


# In[ ]:


rtn_m = df.groupby('timestamp')["y"].mean() # the market return 
vol_m =  df.groupby('timestamp')["y"].std() # the market return volatility cross section
sharp_m = rtn_m/vol_m # sharp ratio
num_m = df.groupby('timestamp')["y"].count() # support number 


# In[ ]:


sns.tsplot(rtn_m)


# 

# In[ ]:


import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


# In[ ]:


def make_corelation(dta,lags):
    fig = plt.figure(figsize=(12,8))
    ax1=fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(dta,lags=lags,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(dta,lags=lags,ax=ax2)


# In[ ]:


make_corelation(rtn_m.values,20)


# In[ ]:


sns.distplot(rtn_m)


# 

# In[ ]:


sns.tsplot(vol_m)


# In[ ]:


make_corelation(vol_m.values,20)


# 

# In[ ]:


sns.distplot(vol_m)


# In[ ]:


sns.tsplot(sharp_m)


# 

# In[ ]:


make_corelation(sharp_m.values,20)


# In[ ]:


sns.distplot(sharp_m)


# In[ ]:


sns.tsplot(num_m)


# 
