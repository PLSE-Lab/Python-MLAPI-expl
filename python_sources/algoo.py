#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


inproceedings = pd.read_csv("../input/output_inproceedings.csv",sep=';',low_memory=False).fillna('')


# In[3]:


articles = pd.read_csv("../input/output_article.csv",sep=';',low_memory=False).fillna('')


# In[18]:


keys_articles = articles[['author','key']]


# In[34]:


keys_inproceedings = inproceedings[['author','key']]


# In[82]:


a = keys_inproceedings['key']
b = (a.str.find('/'))
c = a.str.slice(2)


# In[83]:


print (c.head())


# In[57]:


a = keys_inproceedings['key'].fillna('').astype(str)
a = a.str[keys_inproceedings['key'].str.find('/'):]
print (a.head())
selected_conferences = a.loc[
        a.str.startswith("cvpr") == True or
        a.str.startswith("ecc") == True or
       a.str.startswith("iccv") == True or
        a.str.startswith("ecc") == True or
        a.str.startswith("infocom") == True or
        a.str.startswith("www") == True or
        a.str.startswith("sigcomm") == True or
        a.str.startswith("icse") == True or
        a.str.startswith("asplos") == True or
       a.str.startswith("pldi") == True
]

