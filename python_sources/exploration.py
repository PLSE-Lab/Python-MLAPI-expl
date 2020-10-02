#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk


# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Dataset exploration
# 
# Exploring dataset using some natural language processing techniques.

# In[ ]:


df = pd.read_csv("../input/Reviews.csv", index_col = 0)
print("Number of reviews = {}".format(len(df)))


# In[ ]:


df.head()


# In[ ]:


df.ProductId.value_counts().head()


# In[ ]:


df[df.ProductId=='B007JFMH8M'][['Score','Summary','Text']].head()


# In[ ]:


df.ix[1]


# In[ ]:



df.Score.value_counts()


# In[ ]:


df['datetime'] = pd.to_datetime(df["Time"], unit='s')
df.groupby([df.datetime.dt.year, df.datetime.dt.month]).count()['ProductId'].plot(kind="bar",figsize=(30,10))


# ## Preprocessing
# 
# First we clean up the data.  There are two bodies of text -- the summary and the detailed comment.
# We clean each of these two bodies in turn.

# In[ ]:


# Start with the summary data
df.Summary = df.Summary.astype(str)
downcase = lambda x: x.lower()
df.Summary.apply(downcase)


# In[ ]:


df.Summary = df.Summary.astype(str)
df.Summary.ix[3]


# In[ ]:


df.dtypes


# In[ ]:




