#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


playStore = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
userReviews = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")


# Get first 10 rows of the Input data

# In[ ]:


playStore.head(10)


# Get an info of all the features in the dataset

# In[ ]:


playStore.info()


# Drop all na values

# In[ ]:


playStore = playStore.dropna()


# Get the total count of each unique 'Type' feature

# In[ ]:


playStore['Type'].value_counts()


# Solution

# In[ ]:


print("Total number of free apps are "+ str(playStore['Type'].value_counts()['Free']))
print("Total number of paid apps are "+ str(playStore['Type'].value_counts()['Paid']))


# In[ ]:




