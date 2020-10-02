#!/usr/bin/env python
# coding: utf-8

# # Data Regulation Example
# **This kernel is created by using a dataset including Hip-Hop songs dedicated to presidential candidates of the U.S.A.** 
# **Main purpose is practicing main data regulation methods and kernel regulation methods such as using markdown like I'm doing right now :)**
# 

# In[ ]:




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


data = pd.read_csv('../input/genius_hip_hop_lyrics.csv',encoding = "ISO-8859-1")
data.head()


# In[ ]:


data.info()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


print(data['sentiment'].value_counts(dropna=False))


# In[ ]:


data.describe()
data.head()


# In[ ]:


data.boxplot(column='album_release_date',by = 'sentiment') #Lower quartile/Upper quartile -+ 1.5 IQR(upper quartile - lower quartile) = outliers


# In[ ]:


data_new = data.head()
data_new


# In[ ]:


melted = pd.melt(frame=data_new,id_vars = 'line', value_vars = ['artist','song'])
melted


# In[ ]:


melted.pivot(index = 'line',columns = 'variable',values = 'value')


# In[ ]:





# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0,ignore_index = True)
conc_data_row


# In[ ]:


data1 = data['song'].head()
data2 = data['artist'].head()
conc_data_col = pd.concat([data1,data2],axis=1)
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


data['theme'] = data['theme'].astype('object')
data['sentiment'] = data['sentiment'].astype('category')


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data["theme"].value_counts(dropna = False)


# In[ ]:


data1 = data
data1["theme"].dropna(inplace=True) #Inplace means changes are going to be valid for main dataframe


# In[ ]:


assert data["theme"].notnull().all()


# In[ ]:


data["theme"].fillna('empty',inplace = True) # element of dataframe must be an object.


# In[ ]:


assert data["theme"].notnull().all()


# In[ ]:


assert data.theme.dtypes == np.object


# In[ ]:


data.dtypes


# In[ ]:




