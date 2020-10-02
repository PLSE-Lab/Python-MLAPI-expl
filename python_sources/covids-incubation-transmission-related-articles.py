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


# ## Loading dataset

# We will browse the bulk of comprehensive metadata files of COVID-19 research articles

# In[ ]:


df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
df.shape


# In[ ]:


df.head()


# In[ ]:


df.tail()


# # Coronavirus and Covid-19 transmission-related articles
# 
# Easiest way, scraping throgh the title whether it contained transmission or transmit.
# 
# Pros: this method is the easiest way to found focused article. 
# 
# Cons: Cons: Some articles in the tail from medrxiv didn't list the title, and also some articles could be containing the information but didn't include the keyword on title.

# In[ ]:


title = df.copy()


# We will proceed to the titled articles, and save the medrxiv articles for the next.

# In[ ]:


title = title.dropna(subset=['title'])


# In[ ]:


title.head()


# ### Remove punctuation and turn to lowercase

# In[ ]:


title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)
title['title'] = title['title'].str.lower()


# In[ ]:


title.tail()


# In[ ]:


title['keyword_transmission'] = title['title'].str.find('transmission') 


# If the result prompt `-1`, then the title doesn't contained the keyword. 

# In[ ]:


title.head()


# In[ ]:


included_transmission = title.loc[title['keyword_transmission'] != -1]
included_transmission


# Whoa, there are as much as 624 articles containing transmission!

# In[ ]:


title['keyword_transmit'] = title['title'].str.find('transmit')
included_transmit = title.loc[title['keyword_transmit'] != -1]
included_transmit


# # Coronavirus and Covid-19 incubation-related articles
# 
# Easiest way, scraping throgh the title whether it contained incubation.
# 
# Pros: this method is the easiest way to found focused article. 
# 
# Cons: Some articles in the tail from medrxiv didn't list the title, and also some articles could be containing the information but didn't include the keyword on title.

# In[ ]:


title['keyword_incubation'] = title['title'].str.find('incubation')
included_incubation = title.loc[title['keyword_incubation'] != -1]
included_incubation


# # Searching from Biorxiv (TBA)
# 
# Thanks for the notebook about CORD-19: EDA, parse JSON and generate clean CSV by xhlulu. A specific documentation could be found here https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv . 
# 
# 

# ## I hope this notebook could help for another research related to incubation and transmission, primarily on find the articles. 
