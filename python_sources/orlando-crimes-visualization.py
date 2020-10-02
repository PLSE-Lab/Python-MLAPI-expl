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


data = pd.read_csv('../input/OPD_Crimes.csv')
data.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

data.isna().sum()


# In[ ]:


data.drop(['Case Date Time'], axis=1, inplace=True)
data.drop(['Location'], axis=1, inplace=True)
data.drop(['Orlando Main Street Program Area'], axis=1, inplace=True)
data.drop(['Orlando Commissioner Districts'], axis=1, inplace=True)
data.drop(['Orlando Neighborhoods'], axis=1, inplace=True)
data.drop(['Case Number'], axis=1, inplace=True)

data.head()


# In[ ]:


df = data.copy(deep=1)
df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df['Case Offense Location Type'].value_counts().plot(kind='bar')
plt.title('Location Type of Crime Committed')
plt.yscale('symlog')

fig = plt.gcf()
fig.set_size_inches(17, 7)


# The bar graph above tells us that the Apartment/Condo has been the most favourite spot for crime in Orlando while the Camera Store/Photomat being the least favourite of the lot.

# In[ ]:


df['Case Offense Charge Type'].value_counts().plot(kind='bar')
plt.title('Charge')
plt.yscale('symlog')

fig = plt.gcf()
fig.set_size_inches(17, 7)


# Charge Committed > **10** * Charge Attempted

# In[ ]:


x = []
x.append(df['Case Offense Charge Type'].value_counts()[0])
x.append(df['Case Offense Charge Type'].value_counts()[1])
x
pielabels = 'COMMITTED','ATTEMPTED'


# In[ ]:


plt.pie(x, labels=pielabels, autopct='%1.2f%%')

plt.axis('equal')
plt.show()


# In[ ]:


c = []
df['Case Disposition'].value_counts()


# In[ ]:


c1 = df['Case Disposition'].value_counts()[0]
c2 = df['Case Disposition'].value_counts()[1]
c3 = df['Case Disposition'].value_counts()[2]
c4 = df['Case Disposition'].value_counts()[3]
explode = [0,0,0,0.5]
l = [c1,c2,c3,c4]
labels = 'Closed','Arrest','Inactive','Open'


# In[ ]:


plt.pie(l, labels=labels, autopct='%1.2f%%', explode=explode)

plt.axis('equal')
plt.show()


# Majority of the cases have been **closed**. While almost 5% of the cases remain inactive, less than 2% of the cases are still pending.

# In[ ]:


s = df['Case Offense Category'].value_counts()


# In[ ]:


s


# In[ ]:


s.plot(kind='barh')
plt.title('Case Offense Categories')

fig = plt.gcf()
fig.set_size_inches(10, 5)


# **Theft** dominates the total offenses that were committed in all the cases by almost 4 times the second most committed crime i.e. **Burglary**.

# In[ ]:


df['Case Offense Type'].value_counts()


# In[ ]:


df['Case Offense Type'].value_counts().plot(kind='bar')
plt.title('Charge')
plt.yscale('symlog')

fig = plt.gcf()
fig.set_size_inches(17, 7)


# Larceny has occurred more than 10000 times than incidents of wire frauds have occurred. It is essential to note that Burglary and Theft are among the top 3 contenders with neck-to-neck values of their incidents. 

# In[ ]:




