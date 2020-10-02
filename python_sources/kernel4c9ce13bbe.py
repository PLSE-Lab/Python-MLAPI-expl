#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import os


# In[31]:


filename="../input/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"


# In[32]:


df=pd.read_csv(filename,usecols=['Timestamp','High','Low'],sep=',')


# In[33]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'],unit='s')
df['date'] = df['Timestamp'].dt.date
df['year'] = df['Timestamp'].dt.year
df = df[df.date.notnull() & df.High.notnull() & df.Low.notnull()]


# In[34]:


df.head()


# In[6]:


#Find max drawdown for each day
df1=df.groupby(
   ['date']
).agg(
    {
         'High':max,    
         'Low':min  
    }
)


# In[35]:


df1['max_drawdown']=df1['High']-df1['Low']
df1['Percent drawdown']=(((df1['High']-df1['Low'])/df1['Low'].abs())*100).round(2)


# In[36]:


#MAX DAILY DRAWDOWN FOR ALL AVAILABLE DATA
idx = df1.groupby(['date'])['max_drawdown'].transform(max) == df1['max_drawdown']
df1[idx].max()


# In[37]:


df2=df.groupby(
   ['date','year']
).agg(
    {
         'High':max,   
         'Low':min  
    }
)


# In[38]:


df2['max_drawdown']=df2['High']-df2['Low']
df2['Percent drawdown']=(((df2['High']-df2['Low'])/df2['Low'].abs())*100).round(2)


# In[39]:


#MAX DAILY MAX DRAWDOWNS FOR EACH YEAR
idx2 = df2.groupby(['year'])['max_drawdown'].transform(max) == df2['max_drawdown']
df2[idx2]


# In[46]:


#Find max drawdown for each day
df3=df.groupby(
   ['date','year']
).agg(
    {
         'High':max,    
         'Low':min  
    }
)


# In[47]:


df3['max_drawdown']=df3['High']-df3['Low']
df3['Percent drawdown']=(((df3['High']-df3['Low'])/df3['Low'].abs())*100).round(2)


# In[56]:


#ANNUAL SUMMATION OF DAILY MAX DRAWDOWNS
df3.groupby(['year'])['max_drawdown'].agg('sum')

