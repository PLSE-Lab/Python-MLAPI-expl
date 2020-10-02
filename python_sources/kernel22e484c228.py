#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


cust_df = pd.read_csv("/kaggle/input/parisprocessed/new_cust_df.csv")
prod_df = pd.read_csv("/kaggle/input/parisproductfinal/product_final.csv")


# In[ ]:


cust_df.head(2)


# In[ ]:


prod_df.head(2)


# In[ ]:


ratings = pd.read_excel("/kaggle/input/parisdata/rewiews808K.xlsx")


# In[ ]:


ratings.shape


# In[ ]:


ratings.head()


# In[ ]:


ratings['text'].iloc[0]


# In[ ]:


ratings['url'].iloc[0]


# In[ ]:


prod_df['url'].iloc[0]


# In[ ]:


dummy = [i for i in prod_df['accords']]
total_accords = ','.join(word for word in dummy)
words = total_accords.split(",")
words_set = set(words)


# In[ ]:


my_list = list(words_set)


# In[ ]:


my_list.sort()


# In[ ]:


my_list


# In[ ]:


prod_df['accords'].iloc[0]


# In[ ]:


samples = prod_df['accords'].iloc[0]


# In[ ]:


samples


# In[ ]:


if "woody" in samples:
    print("hello")


# In[ ]:


def get_accords(x):
    my_list_vals = []
    for i in my_list:
        if i in x:
            my_list_vals.append(1)
        else:
            my_list_vals.append(0)
    return [i for i in my_list_vals]


# In[ ]:


sdummy = prod_df['accords'].apply(lambda x: get_accords(x))


# In[ ]:


prod_df['new_accords'] = sdummy


# In[ ]:


tags = prod_df['new_accords'].apply(pd.Series)
tags.head()


# In[ ]:


tags.columns = my_list


# In[ ]:


prod_df = prod_df.join(tags)


# In[ ]:


prod_df.shape


# In[ ]:


del prod_df['accords']
del prod_df['new_accords']


# In[ ]:


prod_df.to_csv("products_finals_with_accords.csv", index=None)


# In[ ]:




