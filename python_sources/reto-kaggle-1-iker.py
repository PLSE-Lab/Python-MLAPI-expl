#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Any results you write to the current directory are saved as output.


# In[ ]:


ds = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")


# In[ ]:


#Actividad 1
ds.tail(10)


# In[ ]:


#Actividad 2
ds.loc[ds.channel_title == 'The Deal Guy'] # actividad 2


# In[ ]:


#Actividad 3
ds.iloc[5000] #


# In[ ]:


#Actividad 4
ds.loc[ds.likes>= 5000000]


# In[ ]:


#Actividad 5
sum(ds.likes[ds.channel_title == 'iHasCupquake'])


# In[ ]:


#Actividad 6
dh=ds.loc[ds.channel_title == 'iHasCupquake' ]
dh.plot(kind = 'bar', x= 'trending_date' , y= 'likes')


# In[ ]:


#Activida7
sum(ds.comment_count[ds.channel_title == 'The Deal Guy'])

