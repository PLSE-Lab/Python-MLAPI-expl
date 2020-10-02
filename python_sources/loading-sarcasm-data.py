#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ### Add column labels 
# * column names from original data - [key.csv ](http://nlp.cs.princeton.edu/SARC/0.0/key.csv)

# In[7]:


# df = pd.read_csv("../input/train-balanced-sarc.csv.gz",sep="\t",names=["label","comment","author","subreddit","score","ups","downs","date","created_utc","parent_comment"]) # Applies to original data , which lacked headers!

df = pd.read_csv("../input/train-balanced-sarcasm.csv")
print(df.shape)
df.head()


# In[ ]:


# Parse UNIX epoch timestamp as datetime: 
# df.created_utc = pd.to_datetime(df.created_utc,unit="s") # Applies to original data , which had UNIX Epoch timestamp! 
df.created_utc = pd.to_datetime(df.created_utc,infer_datetime_format=True) # Applies to original data , which had UNIX Epoch timestamp! 


# In[ ]:


df.describe()


# In[ ]:


## Nothing interesting over time (Likely due to the data being sampled then downsampled by class)
df.set_index("created_utc")["label"].plot()


# ## From this point - build NLP based models +- metadata to predict sarcasm
# * Hint: the subreddit and topic author is often more informative than the text itself. 

# ### Finally, Save out the data with the column names
# * We'll save it uncompressed, so as to make it easy for Kaggle to read. 
# 

# In[ ]:


df.drop(["date"],axis=1).to_csv("train-balanced-sarcasm.csv.gzip",index=False, compression="gzip")

