#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# data read / import
data1 = pd.read_csv('/kaggle/input/uncover/UNCOVER/WHO/who-situation-reports-covid-19.csv')
data2 = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE_update/HDE/total-covid-19-tests-performed-by-country.csv')


# In[ ]:


# who-situation-reports-covid-19 data info
data1.info()


# In[ ]:


# total-covid-19-tests-performed-by-country data info
data2.info()


# In[ ]:


# who-situation-reports-covid-19 data 
data1.columns


# In[ ]:


# total-covid-19-tests-performed-by-country data 
data2.columns


# In[ ]:


# who-situation-reports-covid-19 data 
data1.corr()


# In[ ]:


# total-covid-19-tests-performed-by-country data 
data2.corr()


# In[ ]:


# correlation map
# who-situation-reports-covid-19 data 
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data1.corr(),annot=True,linewidth=10,fmt='.1f',ax=ax)
plt.show()


# In[ ]:


# correlation map
# total-covid-19-tests-performed-by-country data 
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data2.corr(),annot=True,linewidth=10,fmt='.1f',ax=ax)
plt.show()


# In[ ]:


# who-situation-reports-covid-19 data
data1.head()


# In[ ]:


#line plot 
# new_total_deaths and new_confirmed_cases
data1.new_total_deaths.plot(kind = 'line', color = 'r', label = ' confirmed_cases', linewidth=1,alpha =0.7  ,grid = True, linestyle = ':')
data1.new_confirmed_cases.plot(color = 'b',label = 'total_deaths',linewidth=1,alpha = 0.5,grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel(' new_total_deaths ')
plt.ylabel(' new_confirmed_cases ')
plt.title('new_total_deaths and new_confirmed_cases')
plt.show()


# In[ ]:


# x vote_count , y popularity
data1.plot(kind='scatter',x='new_total_deaths',y='new_confirmed_cases',alpha=0.5,color='red')
plt.xlabel('new_total_deaths')
plt.ylabel('new_confirmed_cases')
plt.title('new_total_deaths Count / new_confirmed_cases Scatter plot')
plt.show()


# In[ ]:


# histogram
# values of total_deaths 
data1.total_deaths.plot(kind = 'hist',bins = 100,figsize = (10,10))
plt.show()


# In[ ]:


# alternative scatter plot
plt.scatter(data1.confirmed_cases,data1.total_deaths,color='red')
plt.show()


# In[ ]:


# total-covid-19-tests-performed-by-country data 
data2.head()


# In[ ]:


# total-covid-19-tests-performed-by-country data 
# line plot 
# total_covid_19_tests and year
data2.total_covid_19_tests.plot(kind = 'line', color = 'r', label = ' total_covid_19_tests', linewidth=1,alpha =0.7  ,grid = True, linestyle = ':')
data2.year.plot(color = 'b',label = 'year',linewidth=1,alpha = 0.5,grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel(' total_covid_19_tests ')
plt.ylabel(' year ')
plt.title('total_covid_19_tests and year')
plt.show()


# In[ ]:


# histogram
# values of total_covid_19_tests 
data2.total_covid_19_tests.plot(kind = 'hist',bins = 100,figsize = (10,10))
plt.show()


# In[ ]:


# total_covid_19_tests bigger than 100000
x = data2['total_covid_19_tests']>100000
data2[x]

