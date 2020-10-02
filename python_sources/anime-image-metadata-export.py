#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


def binarize_cols(df,cols=['rating']):
    for col in cols:
        df[col] = df[col].map({'s':0, 'q':1})
    return df


# In[ ]:


df = pd.read_csv("../input/all_data.csv",usecols=['created_at', 'score', 'sample_width',
       'sample_height', 'preview_url', 'tags']).drop_duplicates().dropna()
df.created_at = pd.to_datetime(df.created_at,unit="s",infer_datetime_format=True)
print(df.shape)
print(df.columns)
df.head()


# In[ ]:


df.score.describe()


# In[ ]:


df.score.quantile(0.999)


# In[ ]:


# df["cat_score"] = pd.cut(df.score, 3, labels=["good", "medium", "bad"])


# In[ ]:


df["score"] = df.score.clip(lower=-1, upper=2)


# In[ ]:


df["score"].value_counts(normalize=True)


# In[ ]:


# downsample majority class -  zero score
df = pd.concat([df.loc[df.score==0].sample(frac=0.5),df.loc[df.score!=0]])
print(df.shape)


# In[ ]:


df["score"].value_counts(normalize=True)


# In[ ]:


df.to_csv("safeBooru_animeTags.csv.gz",index=False,compression="gzip")

