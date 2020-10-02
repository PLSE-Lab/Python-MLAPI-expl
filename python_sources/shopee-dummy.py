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


# **Bismillah**

# 1. load data

# In[ ]:


# import library
import pandas as pd

# read dataset
df_dummy = pd.read_csv('../input/dummy-data/Dummy data.csv')

#bikin df baru buat solusi
df_dummy_new = pd.DataFrame(columns=['id','new_number'])

#kolom new_number berisi nilai id (di df lawas)+2
for nilai in df_dummy['id']:
    df_dummy_new = df_dummy_new.append({'id':nilai,'new_number':nilai+2},ignore_index=True)

#karena mintanya di submission cuma ada 2 kolom (defaultnya ada tambahan 1 kolom di awal, dimana kolom ke-0 kosongan), jadi kolom id di set ke kolom 'id'
df_dummy_new.set_index('id', inplace=True)


# 2. datane sg nyar disave

# In[ ]:


df_dummy_new.to_csv(r'../submission_dummy.csv')

