#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # It is used for data visulazation 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# This code import the data from csv to the data1 variable
data1=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding="ISO-8859-1")


# In[ ]:


# This gives general information about data1
data1.info()


# In[ ]:


# This shows first 10 arrows and all columns. So we can look through data1. 
data1.head(10)


# In[ ]:


# This shows correlation between features as number
data1.corr()


# In[ ]:


# This illustrates correaltion asa heat map.
f,ax = plt.subplots(figsize=(50,50))
sns.heatmap(data1.corr(), annot=True, linewidths=.6, fmt= '.1f', ax=ax)
plt.show()


# In[ ]:


# This code gives information about the names in the columns
data1.columns


# In[ ]:


# This shows the number of terror attacks in countries
data1.country.plot(kind='hist', bins=30, figsize=(10,10))
plt.xlabel("Country")
plt.ylabel("Frekans")
plt.title("Terror Attacks in Countries")
plt.legend()
plt.xlim([0, 1100]) # This arranges the range of X- axis
plt.ylim([0, 52000]) # This arranges the range of Y- axis
plt.show()


# In[ ]:


# This shows the number of terror attacks in years
data1.iyear.plot(kind='hist', bins=30, figsize=(10,10))
plt.xlabel("Year")
plt.ylabel("Frekans")
plt.title("Terror Attacks in Years")
plt.legend()
plt.show()


# In[ ]:


# This codes take the names from data1.country_txt and add them into contries as a list
countries=[]
for i in data1.country_txt:
    countries.append(i)
#print(countries)

    


# In[ ]:


# This codes gives the number of attack in certain countries.
print("Mexico",":",countries.count('Mexico'))
print("Turkey",":",countries.count('Turkey'))
print("Norway",":",countries.count('Norway'))
print("Sweden",":",countries.count('Sweden'))
print("United States",":",countries.count('United States'))
print("United Kingdom",":",countries.count('United Kingdom'))


# In[ ]:





# In[ ]:




