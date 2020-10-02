#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


temp = pd.read_excel('../input/temp.xls')
temp = temp.set_index ('YEAR')
temp.head()


# ## Describing the Dataset

# In[ ]:


temp.describe()


# ## Hottest year (Highest ANNUAL min)

# In[ ]:


temp['ANNUAL'].idxmax()


# ## Visualizing Annual Minimum Temperature over Years

# In[ ]:


x = temp.index
y = temp.ANNUAL

plt.scatter(x,y)
plt.show()


# ## Visualizing Temperatures Rise and Fall  (Mean Temp - Months)

# In[ ]:


mean_months = temp.loc[:,'JAN':'DEC'].mean()


# In[ ]:


print (mean_months)


# In[ ]:


plt.plot(mean_months.index, mean_months)


# ## Finding Hottest Seasons (1901-2017)

# In[ ]:


hottest_seasons = {'Winter' : temp['JAN-FEB'].idxmax(),
                   'Summer' : temp['MAR-MAY'].idxmax(),
                   'Monsoon': temp['JUN-SEP'].idxmax(),
                   'Autumn' : temp['OCT-DEC'].idxmax()}
print (hottest_seasons)


# ## Finding  the Most Extreme Year
# 1. Calculate min()  and max() on JAN to DEC columns for each row
# 2. Calculate difference = max - min for each row
# 3. Add difference (DIFF) column to the dataframe
# 4. Do idxmax() on DIFF column

# In[ ]:


temp ['DIFF'] = temp.loc[:,'JAN':'DEC'].max(axis=1) - temp.loc[:,'JAN':'DEC'].min(axis=1)
temp.DIFF.idxmax()


# In[ ]:


axes= plt.axes()
axes.set_ylim([5,15])
axes.set_xlim([1901,2017])
plt.plot(temp.index, temp.DIFF)

temp.DIFF.mean()


# ## Looking into Winter Abnormalities

# Get temperature data for 12 months

# In[ ]:


year_dict = temp.loc[:,'JAN':'DEC'].to_dict(orient='index')


# Let's assume that four coldest months are winter months

# In[ ]:


sorted_months = []
for key, value in year_dict.items():
    sorted_months.append (sorted(value, key=value.get)[:4]) #Only take first 4 elements out


# In[ ]:


winter = sorted_months[:]
winter_set = []
for x in winter:
    winter_set.append (set(x))
temp['WINTER'] = winter_set


# Finding most frequent months i.e. regular winter months

# In[ ]:


winter_routine = max(sorted_months, key=sorted_months.count)


# Finally, finding out irregular winters!
# **As we can see years 1957, 1976, 1978, 1979 had winters upto March**

# In[ ]:


temp.WINTER [temp.WINTER != set(winter_routine)]


# In[ ]:




