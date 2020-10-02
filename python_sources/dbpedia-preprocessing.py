#!/usr/bin/env python
# coding: utf-8

# #### Original data cleaning kernel
# 

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv("../input/DBP_wiki_data.csv",usecols=['text', 'l1', 'l2', 'l3', 'wiki_name']).drop_duplicates(subset="text")

print(df.shape)
print(df.nunique())
df.head()


# In[ ]:


df.columns


# ## Text length analysis
# * Drop if excessive (super short or long) 
# * For SB - will need to truncate or drop manually . For deep learning: padding/truncating i s part of the data prep regardless. 
# * Relevant for computational load, 

# In[ ]:


df["word_count"] = df.text.str.split().str.len()
print(df["word_count"].quantile([0.02,0.98]))
df["word_count"].describe()


# In[ ]:


print(df["word_count"].quantile([0.01,0.03,0.98,0.99]))
df["word_count"].describe()


# ### rows with very few words look like data errors.
# * We'll Drop them

# In[ ]:


df.loc[df["word_count"]<5]


# In[ ]:


print("Old shape",df.shape[0])
df = df.loc[(df["word_count"]>10) & (df["word_count"]<500)]
print("New shape",df.shape[0])

df["word_count"].describe()


# #### drop some cols
# * Note we'll still need to drop class columns (but htese can vary per task)

# In[ ]:


# no need to export word count col. 
# WE also drop the wiki page name
df.drop(["word_count","wiki_name"],axis=1,inplace=True)


# ## Rare class frequencies:
# * from 2,700 to 180. 
# * No need to remove the rarest classes? (180+ is reasonable. Rare but reasonable). 
# * Data is clearly very imbalanced

# In[ ]:


df.l3.value_counts()


# In[ ]:


df.l2.value_counts()


# ## Create train/test/validation split partition
# * Stratify by classes

# In[ ]:


print("orig",df.shape)
df_train,df_test = train_test_split(df, test_size=0.18, random_state=42,stratify = df["l3"])

print("Train size",df_train.shape, "\n Test size",df_test.shape)


# In[ ]:


## validation set split:

df_train,df_val = train_test_split(df_train, test_size=0.13, random_state=42,stratify = df_train["l3"])
print("Final Train size",df_train.shape, "\nValidation size",df_val.shape)


# In[ ]:


df_train.to_csv("DBPEDIA_train_v1.csv.gz",index=False,compression="gzip")
df_val.to_csv("DBPEDIA_val_v1.csv.gz",index=False,compression="gzip")
df_test.to_csv("DBPEDIA_test_v1.csv.gz",index=False,compression="gzip")

