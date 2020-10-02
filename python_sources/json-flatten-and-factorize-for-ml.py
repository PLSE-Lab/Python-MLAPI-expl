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


# Several columns in the dataset come is as JSON formats.  For the purpose of ML, we can process the JSON strings and then factorize the columns for further analysis.  
# Information from these two locations are very helpful;
# **Step 1: Flatten**
# https://www.kaggle.com/rajuspartan/exploratory-data-analysis-with-reusable-functions
# #Flatening JSON columns
# **Step 2: Factorize**
# https://stackoverflow.com/questions/50394099/separate-columns-based-on-genre
# #featurize the genre column
# 

# In[ ]:


# Read the csv file into pandas dataframe
train = pd.read_csv('../input/train.csv')
new_genres = pd.DataFrame(train['genres'])
new_genres.head()


# In[ ]:


#Flattening JSON columns
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

new_genres['genres'] = new_genres.genres.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
new_genres.head()


# In[ ]:


#featurize the genre column
new_genres = new_genres['genres'].str.get_dummies(',')
print(new_genres)


# In[ ]:


#add genres back to data (join)
train = pd.concat([train, new_genres], axis = 1, sort = False)
train.head()


# In[ ]:


#Happy Machine Learning!


# In[ ]:





# In[ ]:





# In[ ]:




