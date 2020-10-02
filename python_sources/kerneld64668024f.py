#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from datetime import datetime


# In[9]:


c1 = pd.read_csv('../input/c1.csv')
c1 = c1.dropna(axis=0, how='any')


# In[10]:


c1['month'] = '9'
c1['year'] = '2017'


# In[11]:


c1['START'] = c1.VAR4.str.split(':',1,expand=True)[1]
c1['day'] = c1.VAR4.str.split('P',1,expand=True)[1].str.split(':',1,expand=True)[0]
c1['hour'] = c1.START.str.split(':',expand=True)[0]
c1['minute'] = c1.START.str.split(':',expand=True)[1]
c1['second'] = c1.START.str.split(':',expand=True)[2]
c1['starttime'] = pd.to_datetime(c1[['year','month','day','hour','minute','second']])


# In[12]:


c1['END'] = c1.VAR7.str.split(':',1,expand=True)[1]
c1['day'] = c1.VAR7.str.split('P',1,expand=True)[1].str.split(':',1,expand=True)[0]
c1['hour'] = c1.END.str.split(':',expand=True)[0]
c1['minute'] = c1.END.str.split(':',expand=True)[1]
c1['second'] = c1.END.str.split(':',expand=True)[2]
c1['endtime'] = pd.to_datetime(c1[['year','month','day','hour','minute','second']])


# In[13]:


c1['seconds'] = (c1['endtime']-c1['starttime']) / np.timedelta64(1, 's')
c1.head(5)


# In[14]:


c1.drop(c1[c1.seconds<300].index, inplace=True)
c1.head(5)


# In[23]:


c1.drop(c1[c1.VAR8=='null'].index, inplace=True)
c1.drop(c1[c1.VAR9=='null'].index, inplace=True)
c1.VAR8 = c1.VAR8.astype('float64')  
c1.VAR9 = c1.VAR9.astype('float64')  


# In[28]:


c1['5min_long'] = c1['VAR5']+(c1['VAR8']-c1['VAR5'])*300/c1['seconds']
c1['5min_lat'] = c1['VAR6']+(c1['VAR9']-c1['VAR6'])*300/c1['seconds']


# In[38]:


c1['10min_long'] = c1.loc[(c1.seconds>=600),'VAR5'] + (c1.loc[(c1.seconds>=600),'VAR8']-c1.loc[(c1.seconds>=600),'VAR5'])*600/c1['seconds']
c1['10min_lat'] = c1.loc[(c1.seconds>=600),'VAR6'] + (c1.loc[(c1.seconds>=600),'VAR9']-c1.loc[(c1.seconds>=600),'VAR6'])*600/c1['seconds']


# In[36]:


c1['15min_long'] = c1.loc[(c1.seconds>=900),'VAR5'] + (c1.loc[(c1.seconds>=900),'VAR8']-c1.loc[(c1.seconds>=900),'VAR5'])*900/c1['seconds']
c1['15min_lat'] = c1.loc[(c1.seconds>=900),'VAR6'] + (c1.loc[(c1.seconds>=900),'VAR9']-c1.loc[(c1.seconds>=900),'VAR6'])*900/c1['seconds']


# In[40]:


c1['30min_long'] = c1.loc[(c1.seconds>=1800),'VAR5'] + (c1.loc[(c1.seconds>=1800),'VAR8']-c1.loc[(c1.seconds>=1800),'VAR5'])*1800/c1['seconds']
c1['30min_lat'] = c1.loc[(c1.seconds>=1800),'VAR6'] + (c1.loc[(c1.seconds>=1800),'VAR9']-c1.loc[(c1.seconds>=1800),'VAR6'])*1800/c1['seconds']


# In[44]:


c1.to_csv('c1_newcoordinate.csv')

