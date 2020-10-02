#!/usr/bin/env python
# coding: utf-8

# In[103]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[104]:


df = pd.read_csv("../input/unigram_freq.csv")
df.head(10)


# **Top 10 Most Used Words**
# 
# As the data has been sorted by the word count, let's plot the first 10 data

# In[105]:


top10 = df.iloc[0:10]
plt.figure(figsize=(10,6))
sns.barplot("word", "count", data=top10, palette="Blues_d").set_title("Top 10 Words")


# **Top 10 Least Used Words**

# In[106]:


df.sort_values(by="count").iloc[0:10]


# Wait, what? Are those even english? And they are apparently have the same count

# **Top 10 Longest Words**
# 
# Let's see what are the 10 longest words

# In[107]:


s = df.word.str.len().sort_values(ascending=False).index
longest10 = df.reindex(s).iloc[0:10]
plt.figure(figsize=(10,6))
sns.barplot("count", "word", data=longest10, orient="h", palette="Blues_d").set_title("Top 10 Longest Words")


# Well I didn't even know that those words exist and there are actually a quite number of people used them.

# **Alphabets as Words**
# 
# Next one, let's see the usage of each alphabet as an individual word

# In[108]:


alphabet = df.reindex(s).iloc[::-1][2:28].sort_values(by="count", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot("word", "count", data=alphabet, palette="Blues_d").set_title("Alphabets")

