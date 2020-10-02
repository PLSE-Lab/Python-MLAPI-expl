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


df = pd.read_csv("../input/17k-apple-app-store-strategy-games/appstore_games.csv")


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


df.drop(['ID', 'URL', 'Subtitle', 'Icon URL', 'Description'], inplace=True, axis=1)


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.shape


# We can group by Developer to see which company is the most popular one

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import operator


# In[ ]:


df.dtypes


# We can group by developers and calculate their total user rating scores

# In[ ]:


developers = []
dev_user_rating = []
for dev, group in df.groupby('Developer'):

    dev_user_rating.append((dev, np.sum(group['Average User Rating'] * group['User Rating Count'])))

dev_user_rating.sort(key = operator.itemgetter(1), reverse = True)
top_10_devs = dev_user_rating[:10]
print (top_10_devs)


# In[ ]:


dev, total_rating = list(zip(*top_10_devs))


# Now we can plot the most popular 10 developers in app store

# In[ ]:


plt.figure(figsize=(32, 12))
sns.barplot(x=list(dev), y=list(total_rating), palette="rocket")
plt.show()

