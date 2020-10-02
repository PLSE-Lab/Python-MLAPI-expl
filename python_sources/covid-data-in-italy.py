#!/usr/bin/env python
# coding: utf-8

# # Analysis of Corona Virus Data Italy

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Reading data with csv reader & print header of provinces

# In[ ]:


province_data = pd.read_csv('../input/covid19-in-italy/covid19_italy_province.csv', index_col='Date', parse_dates=True)
region_data = pd.read_csv('../input/covid19-in-italy/covid19_italy_region.csv', index_col='Date', parse_dates=True)
#region_data = pd.read_csv('../input/covid19-in-italy/covid19_italy_region.csv')


#region_data['MyDate'] =  pd.to_datetime(region_data['Date']).dt.strftime('%m-%d')
#region_data.set_index('MyDate')
#print ("Region Data")
region_data.head()


# In[ ]:


print ("Region Data")
region_data.head()


# We can now view the columns

# Check tail of code

# In[ ]:


print ("Region Data (tail)")
region_data.tail()


# In[ ]:


for (idx,column) in enumerate(region_data.columns):
    print("%d) %s" % (idx,column) )


# Listing regions

# In[ ]:


region_columns = region_data['RegionName']
for (idx, region) in enumerate(region_columns.unique()):
    print("%d) %s" % (idx,region) )


# Comparison of cases Lombardia/Emilia/Veneto

# In[ ]:


#Define target region list
plt.figure(figsize=(10,6))
tgt_region_l = ['Emilia-Romagna', 'Lombardia', 'Veneto']
emilia_data = region_data.loc[region_data.RegionName == 'Emilia-Romagna']
lombardia_data = region_data.loc[region_data.RegionName == 'Lombardia']
veneto_data = region_data.loc[region_data.RegionName == 'Veneto']
for tgt_region in tgt_region_l:
    my_data = region_data.loc[region_data.RegionName ==tgt_region]
    ax = sns.lineplot(data=my_data["CurrentPositiveCases"], label=tgt_region)


# In[ ]:


Same regions : new cases


# In[ ]:



plt.figure(figsize=(14,6))
for tgt_region in tgt_region_l:
    my_data = region_data.loc[region_data.RegionName ==tgt_region]
    ax = sns.lineplot(data=my_data["NewPositiveCases"], label=tgt_region)
    


# New Cases versus tests

# In[ ]:


region_data["TestCasesRatio"] = region_data['NewPositiveCases']/region_data['TestsPerformed']
plt.figure(figsize=(14,6))
for tgt_region in tgt_region_l:
    my_data = region_data.loc[region_data.RegionName ==tgt_region]
    ax = sns.lineplot(data=my_data["TestCasesRatio"], label=tgt_region)


# Grahpic on hospitalized patients

# In[ ]:



plt.figure(figsize=(14,6))
for tgt_region in tgt_region_l:
    my_data = region_data.loc[region_data.RegionName == tgt_region]
    sns.lineplot(data=my_data["IntensiveCarePatients"], label="%s: " % tgt_region)


# Intensive care patients

# In[ ]:


plt.figure(figsize=(14,6))

for tgt_region in tgt_region_l:
    my_data = region_data.loc[region_data.RegionName == tgt_region]
    sns.lineplot(data=my_data["HospitalizedPatients"], label="%s: Hospitalized patients" % tgt_region)


# Barplot of cumulative data

# In[ ]:


# Get last day
plt.figure(figsize=(14,6))
last_day_data = region_data.loc[region_data.index[-1]]
ax = sns.barplot(x="RegionName", y="CurrentPositiveCases", data=last_day_data)
y = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# 
