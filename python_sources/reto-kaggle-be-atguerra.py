#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

ds = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")


# In[ ]:


#Actividad 1
ds.tail(10)


# In[ ]:


#Actividad 2
ds.loc[ds.channel_title == 'The Deal Guy']


# In[ ]:


#Actividad 3
ds.iloc[5000]


# In[ ]:


#Actividad 4
ds.loc[ds.likes >= 5000000]


# In[ ]:


#Actividad 5
sum(ds.likes[ds.channel_title == 'iHasCupquake'])


# In[ ]:


#Actividad 6
ds.loc[ds.channel_title == 'zefrank1'].plot(kind='bar', x='views', y='likes')


# In[ ]:


#Actividad 6 original
ds.loc[ds.channel_title == 'iHasCupquake'].plot(kind='bar', x='trending_date', y='likes')

