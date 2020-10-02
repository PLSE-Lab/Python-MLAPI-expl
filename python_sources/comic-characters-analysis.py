#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df_dc = pd.read_csv('../input/dc-wikia-data.csv')
df_marvel = pd.read_csv('../input/marvel-wikia-data.csv')


# In[ ]:


df_dc.head()


# In[ ]:


df_marvel.head()


# In[ ]:


df_dc.shape


# In[ ]:


df_marvel.shape


# In[ ]:


p = sns.countplot(x="ID", data=df_dc)
_ = plt.title("DC")


# In[ ]:


p = sns.countplot(x="ID", data=df_marvel)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")


# In[ ]:


p = sns.countplot(x="ALIGN", data=df_dc)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("DC")


# In[ ]:


p = sns.countplot(x="ALIGN", data=df_marvel)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")


# In[ ]:


p = sns.countplot(x="EYE", data=df_dc)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("DC")


# In[ ]:


p = sns.countplot(x="EYE", data=df_marvel)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")


# In[ ]:


p = sns.countplot(x="HAIR", data=df_dc)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("DC")


# In[ ]:


p = sns.countplot(x="HAIR", data=df_marvel)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")


# In[ ]:


p = sns.countplot(x="SEX", data=df_dc)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("DC")


# In[ ]:


p = sns.countplot(x="SEX", data=df_marvel)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")


# In[ ]:


p = sns.countplot(x="ALIVE", data=df_dc)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("DC")


# In[ ]:


p = sns.countplot(x="ALIVE", data=df_marvel)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")


# In[ ]:


p = sns.heatmap(df_dc.corr(), annot=True)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("DC")


# In[ ]:


p = sns.heatmap(df_marvel.corr(), annot=True)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")


# In[ ]:


dc_sort = df_dc.sort_values("APPEARANCES", ascending=False)
dc_top_10 = dc_sort.head(10)
dc_top_10


# In[ ]:


marvel_sort = df_marvel.sort_values("APPEARANCES", ascending=False)
marvel_top_10 = marvel_sort.head(10)
marvel_top_10


# In[ ]:


p = sns.barplot(x="name", y="APPEARANCES", data=dc_top_10)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("DC")


# In[ ]:


p = sns.barplot(x="name", y="APPEARANCES", data=marvel_top_10)
_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.title("Marvel")

