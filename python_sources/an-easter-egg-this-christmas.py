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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Trip takes longer if every 10th step doesn't come from a prime, so it's important to be able to find the prime cities.

# In[ ]:


def is_prime(n):
    """Determines if a positive integer is prime."""

    if n > 2:
        i = 2
        while i ** 2 <= n:
            if n % i:
                i += 1
            else:
                return False
    elif n != 2:
        return False
    return True


# In[ ]:


df = pd.read_csv("../input/cities.csv")


# In[ ]:


df.head()


# In[ ]:


df['Prime'] = df['CityId'].apply(is_prime)


# In[ ]:


fig = plt.figure(figsize=(18,18))
plt.scatter(df.X, df.Y, c=df['Prime'], marker=".", alpha=.5);


# In[ ]:


sub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sub.head()


# In[ ]:


df['xplusy'] = df.X + df.Y


# In[ ]:


df.sort_values('xplusy')['CityId'].values


# In[ ]:


sub['Path'] = np.append(0, df.sort_values('xplusy')['CityId'].values)


# Ensure that path ends at North Pole.

# In[ ]:


sub.loc[24740] = 197338


# In[ ]:


sub.loc[197769] = 0


# In[ ]:


sub.to_csv('simple_submission.csv', index=False)


# In[ ]:




