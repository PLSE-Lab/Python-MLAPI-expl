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


import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/chopsticks-1992/chopstick-effectiveness.csv")


# In[ ]:


data.head(10)


# In[ ]:


data.describe()


# In[ ]:


data.info()


# **Dataset Description**
# 
# In this dataset we have 3 columns, Food Pinching Efficency which is the one we will use to determine if a chopstick length is better or worse, Indiviual which is not told the exact meaning but i think is the person that was used in the test, and Chopstick Length which is just the chopstick length.

# In[ ]:


data = data.drop('Individual', axis = 1)


# In[ ]:


data.head()


# In[ ]:


data.rename(columns = {'Food.Pinching.Efficiency': 'efficiency', 'Chopstick.Length': 'len_chop'}, inplace=True)
data.head()


# In[ ]:


plt.plot(data['len_chop'], data['efficiency'])
plt.xlabel("Length")
plt.ylabel("Efficiency")
plt.show()

#Highest efficiency is found when length is 240


# In[ ]:


# There are six length
plt.hist(data['len_chop'])


# In[ ]:


data.nunique()


# In[ ]:


plt.hist(data['efficiency'])
plt.show()


# In[ ]:


# Check correlation between features
sns.pairplot(data)


# **Highest efficiency is found when length is 240**

# In[ ]:




