#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Building dataframe from scracth
# zip()
isim = ['Melih','Ahmet','Fatma','Ece','Serkan']
yas = [32,28,30,25,20]
list_label = ['AD','YAS']
list_col = [isim, yas]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


# In[ ]:


# Adding new column
#df['SOYAD'] = ['Akdag', 'Kemal', 'Ozturk', 'Bilir', 'Kaya']
#df


# In[ ]:


# Adding new column into specific location
df.insert(1, "SOYAD", ['Akdag', 'Kemal', 'Ozturk', 'Bilir', 'Kaya'])
df


# In[ ]:


# Broadcasting
df['MAAS'] = 1000 
df


# In[ ]:


# Indexing Pandas Time Series
t_list = ['2014.02.12','2016.03.16','2018.02.05','2017.03.04','2014.12.25'] # type is string
datetime_object = pd.to_datetime(t_list)
df['BASLANGIC'] = datetime_object
df


# In[ ]:


# Indexing according to date
df = df.set_index('BASLANGIC')
df


# In[ ]:


# To search a data according to index (which is date in this example)
print(df.loc['2014-12-25'])


# In[ ]:


# Searching between dates
print(df.loc['2014-01-01':'2017-01-01'])


# In[ ]:


# Resampling Pandas Timeseries
df.resample('A').mean()  # 


# In[ ]:


df.resample('M').mean()


# In[ ]:


df.resample('M').first().interpolate('linear')


# In[ ]:


df.resample('M').mean().interpolate('linear')

