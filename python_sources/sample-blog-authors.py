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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/blogtext.csv")
print(df.shape)
df.head()


# In[ ]:


df.drop_duplicates(subset="text",inplace=True)


# In[ ]:


df.text.str.len().describe()


# It appears that we have some outliers with text length.
# Authorship of consistantly long/short posts is somewhat likely to be an interesting feature for author/demographic classification.
# * Still, we can truncate some of the more extreme cases 

# In[ ]:


df.text.str.len().plot()


# In[ ]:


df = df.loc[(df.text.str.len() < 18000) & (df.text.str.len() > 7)]


# In[ ]:


df.text.str.len().plot()


# In[ ]:


df.loc[df.text.str.len() < 30].shape


# In[ ]:


df.describe()


# In[ ]:


df.age.value_counts()


# In[ ]:


df.topic.value_counts()


# In[ ]:


df["word_count"] = df.text.str.split().str.len()


# In[ ]:


df["char_length"] = df.text.str.len()


# In[ ]:


df["id_count"] = df.groupby("id")["id"].transform("count")


# In[ ]:


df.head(12)


# In[ ]:


df.date = pd.to_datetime(df.date,errors="coerce",infer_datetime_format=True)


# In[ ]:


df.tail()


# In[ ]:


df.drop_duplicates(subset="id")["id_count"].describe()


# In[ ]:


df.shape


# Filter out super frequent users (otherwise prediction will be biased to them). 

# In[ ]:


df = df.loc[df.id_count < 200]
df.shape


# In[ ]:


df.head()


# In[ ]:


df["age_group"] = pd.qcut(df["age"],3,precision=0,)


# In[ ]:


df.head()


# In[ ]:


df.age_group.value_counts()


# In[ ]:


df["age+sex"] = df["age_group"].astype(str)+df["gender"]


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df["age+sex"].value_counts()


# In[ ]:


# df.to_csv("blogAuthors_sample.csv.gz",index=False,compression="gzip")


# In[ ]:


df.head()


# In[ ]:


df.drop(["age_group"],axis=1).drop_duplicates(subset="id",keep="first").to_csv("blogAuthors_sample_distinct.csv.gz",index=False,compression="gzip")


# In[ ]:


df[["id","date","topic","text","word_count"]].to_csv("blogAuthors_sample_history.csv.gz",index=False,compression="gzip")


# In[ ]:




