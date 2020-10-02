#!/usr/bin/env python
# coding: utf-8

# <h1>Import Data Set </h1>

# In[16]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transliterate import translit, get_available_language_codes

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train_data = pd.read_csv("../input/train.csv")
train_data.head(n=5)

# Any results you write to the current directory are saved as output.


# <h1>Translate the Russian characters into English using transliterate</h1>

# In[29]:


region = (train_data['region']).apply(translit, 'ru', reversed=True)
city = (train_data['city']).apply(translit, 'ru', reversed=True)
parent_category_name = (train_data['parent_category_name']).apply(translit, 'ru', reversed=True)
category_name = (train_data['category_name']).apply(translit, 'ru', reversed=True)

param_1 = train_data['param_1'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True) )
param_2 = train_data['param_2'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True) )
param_3 = train_data['param_3'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True) )
title = train_data['title'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True) )
description = train_data['description'].apply(lambda row:row if pd.isnull(row) else translit(row, 'ru', reversed=True) )


# **Created the translated DataFrame**

# In[40]:


#description.head(n=20)
#pd.unique(train_data['param_1'])
#pd.isnull(train_data['param_1'])
#train_data['param_1'].apply(lambda row: translit(row, 'ru', reversed=True) if row.notnull() else row)
train_data_translated = train_data
train_data_translated['region'] = region
train_data_translated['city'] = city
train_data_translated['parent_category_name'] = parent_category_name
train_data_translated['category_name'] = category_name
train_data_translated['param_1'] = param_1
train_data_translated['param_2'] = param_2
train_data_translated['param_3'] = param_3
train_data_translated['title'] = title
train_data_translated['description'] = description


# **Output the translated DataFrame**

# In[48]:


train_data_translated.head(n=20)
train_data_translated.to_csv("train_translated.csv")


# In[47]:


train_data_translated.head(n=20).to_csv("train_translated2.csv")

