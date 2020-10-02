#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
sns.set()


# In[ ]:


df = pd.read_csv("../input/leaderboard/order-brushing-shopee-code-league-publicleaderboard.csv")


# In[ ]:


sns.distplot(df.Score, bins=100, kde=False)


# In[ ]:


df.groupby('Score').TeamId.count().sort_values(ascending=False).head(20)


# Here `0.22656` is score uploading all zero.
# 
# Except this, 81 teams coincidently share the same answer with score `0.89933`.
