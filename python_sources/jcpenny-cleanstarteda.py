#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv("../input/jcpenney_com-ecommerce_sample.csv").drop(['uniq_id'],axis=1)
df[['list_price', 'sale_price']] = (df[['list_price', 'sale_price']]
        .applymap(lambda v: str(v)[:4]).dropna().astype(np.float64))


# In[ ]:


print(df.shape)
df.head()


# In[ ]:


df.columns


# Only ~ 10K unique items. Duplicates likely due to reviews.

# In[ ]:


len(set(df.name_title))


# * No complete duplicates (despite repeating products) , but many rows with no sale_price and even more without a sale_price!a

# In[ ]:


df.dropna(subset=['list_price', 'sale_price']).drop_duplicates().shape


# In[ ]:


df.dropna(subset=['sale_price']).shape


# In[ ]:


df = df.dropna(subset=['sale_price','list_price']).drop_duplicates(subset=["Reviews","sku","sale_price",'list_price'])
df.shape


# In[ ]:


len(set(df.name_title))


# In[ ]:


df["average_product_rating"] = df["average_product_rating"].str.replace(" out of 5","")


# In[ ]:


df.head()


# In[ ]:


# Note: this should be used only for metadata /feature engineering, not left in either model directly!
df["sale_discount"] =  100*df['sale_price'].div(df['list_price'])
df["sale_discount"] = df["sale_discount"].apply(lambda v: str(v)[:5]).astype(float)


# In[ ]:


df.head()


# ### Split categories.
# * Shamelessly modified from Mercari: https://www.kaggle.com/huguera/mercari-data-analysis
# * This sort of hierarchical categories is very common in these sort of dataset. Having the higher level brands/categories (i.e the mid-levels) may help generalize.

# In[ ]:


def transform_category_name(category_name):
    try:
        main, sub1, sub2= category_name.split('|')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan

df['category_main'], df['category_sub1'], df['category_sub2'] = zip(*df['category_tree'].apply(transform_category_name))


# In[ ]:


df.head()


# ### Count uniques per column - then drop any unaries

# In[ ]:


df.apply(lambda x: len(x.unique()))

## Looks like category isn't the same as the lowest/category_sub2 col!            


# In[ ]:


df.category_main.value_counts()
# the #2 at the top level is simply NaNs..


# In[ ]:


df.drop(["category_main",'category_tree'],axis=1,inplace=True)


# In[ ]:


df.to_csv("jcPenny_subset.csv.gz",index=False,compression="gzip")

