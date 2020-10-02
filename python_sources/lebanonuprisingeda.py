#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pprint import pprint

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# read the json file into a dataframe and display first few rows
df = pd.read_json("/kaggle/input/lebanon-uprising-october-2019-tweets/LebTaxUprising-100k.json",)
df.head()


# In[ ]:


# show basic stats
df.describe()


# In[ ]:


top_tweet = df[df['retweet_count']==4181]['text']
pprint(top_tweet);


# In[ ]:


df['retweet_count'].hist(bins=5);


# In[ ]:


# see language codes (2-letters) here: https://www.sitepoint.com/iso-2-letter-language-codes/
df['lang'].value_counts()


# In[ ]:


# top 10 active users
df['screen_name'].value_counts(sort=True)[:10]


# In[ ]:




