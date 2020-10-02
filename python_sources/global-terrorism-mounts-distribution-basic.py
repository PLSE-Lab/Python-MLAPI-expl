#!/usr/bin/env python
# coding: utf-8

# **Which months are more dangerous**
# 
# I'm new to Kaggle and Python :)
# 
# I'm learning.. ^^
# 
# This data is my target; to understand which months of the year are more targets.
# 
# Dataset : Global Terrorism Database
# 
# Dataset link : [https://www.kaggle.com/START-UMD/gtd](http://www.kaggle.com/START-UMD/gtd)
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Bacis visualization tool.
import seaborn as sns # Cool visualization tools. ^^


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# I didn't understand why, but I got a "utf-8" error here.
# 
# Therefore I was able to retrieve the data with the following method.

# In[ ]:


data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='Windows-1252')

#11.11.2018 
#pd.read_csv('../input/globalterrorismdb_0718dist.csv') 
#I used the above code because I received an error.(utf-8 Error!)


# In[ ]:


data.head(3)


# In[ ]:


data.info()


# In[ ]:


data.columns


# I'm creating a little bit of data for myself because it's too much data for me.

# In[ ]:


globaldata = data[['iyear','imonth','country','country_txt','region','region_txt','city','success','attacktype1','attacktype1_txt']].copy()


# In[ ]:


type(globaldata)


# In[ ]:


globaldata.head(10)


# In[ ]:


January =  globaldata['imonth']<2
globaldata[January].head(5)
#Nice just January..


# In[ ]:


globaldata.imonth.plot(kind = 'hist',bins = 50, figsize = (15,15))
plt.show()

# so how do all months look?


# In fact, as we saw above, we see a little more activity than others in May.

# In[ ]:


globaldata.iyear.plot(kind = 'hist',bins = 100, figsize = (15,15))
plt.show()


# In[ ]:


dropzero = globaldata['imonth'] != 0
globaldata[dropzero].head(5)


# In[ ]:


globaldata[dropzero].describe()


# In[ ]:


year_count = globaldata[dropzero].groupby(['iyear']).count()
month_count = globaldata[dropzero].groupby(['imonth']).count()


plt.plot(month_count,color='red',label = "Monthly Distribution")
plt.title('Monthly Distribution')
plt.xlabel('Mounts')
plt.ylabel('count')
plt.show()

plt.plot(year_count,color='red',label = "Years Distribution")
plt.title('Years Distribution')
plt.xlabel('Year')
plt.ylabel('count')
plt.show()


# Conclusion: May, seems a bit more dangerous, but the months of March and December seem to be safer.
# 
# > Thanks for your time. ^^
