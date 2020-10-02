#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# I have noted the number of teams in last month,Now I did some sample prediction of teams and competitors, The result of teams at **2018-06-21** is **1962**. Maybe the ture number **will be lower,**because there will be inflection point in the end,I think.

# In[3]:


df = pd.read_excel('../input/avitoxlsx/avito.xlsx',parse_dates=['date'])


# In[4]:


df_all = pd.DataFrame({'date':pd.date_range('2018-5-15','2018-6-11')})


# In[5]:


#fill nan
df = df_all.merge(df,on='date',how='left')


# In[6]:


def fillna(vals):
    lastval = 0
    lastnum = 0
    nans = []
    for i,num in enumerate(vals):
        if np.isnan(num):
            nans.append(i)
        if not np.isnan(num) and nans:
            for j in nans:
                vals[j]= (num+lastnum)/2
            nans = []
#         print(lastnum,num,not np.isnan(num))
        if not np.isnan(num):
            lastnum=num
    return vals


# In[7]:


df['teams'] = fillna(df.teams.values)
df['competitors'] = fillna(df.competitors.values)


# In[8]:


df


# In[9]:


df.plot.line(x='date',y='competitors')


# In[10]:


df.plot.line(x='date',y='teams')


# In[11]:


from  sklearn.linear_model import *


# In[79]:


lr = Ridge(alpha=10)


# In[80]:


lr.fit(df.date.dt.dayofyear.values.reshape(-1,1),df.competitors)


# In[81]:


test_df = pd.DataFrame({'date':pd.date_range(start='2018-06-11',end='2018-06-21'),'teams':np.NAN,'competitors':np.NAN})


# In[82]:


test_df


# In[83]:


test_df['competitors'] = lr.predict(test_df.date.dt.dayofyear.values.reshape(-1,1))


# In[84]:


test_df.plot.line(x='date',y='competitors')


# In[85]:


lr.fit(df.date.dt.dayofyear.values.reshape(-1,1),df.teams)


# In[86]:


test_df['teams'] = lr.predict(test_df.date.dt.dayofyear.values.reshape(-1,1))


# In[87]:


test_df.plot.line(x='date',y='teams')


# In[88]:


print('teams:',test_df.teams.iloc[-1],'competitimes:',test_df.competitors.iloc[-1])


# In[89]:


test_df


# In[ ]:




