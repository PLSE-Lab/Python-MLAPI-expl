#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import os
print(os.listdir("../input"))
print(os.listdir("../input/final-subs"))
print(os.listdir("../input/lgbm-tunning-bayes-search"))


# In[ ]:


xgb = pd.read_csv("../input/final-subs/xgb_final.csv")
print(xgb.shape)
lgb = pd.read_csv("../input/lgbm-tunning-bayes-search/clean_submission.csv")
print(lgb.shape)
cat = pd.read_csv("../input/final-subs/cat_boost.csv")
print(cat.shape)

xgb.head()


# In[ ]:


lgb.head()


# In[ ]:


cat.head()


# In[ ]:


comb = cat
comb.rename(columns={'time_to_failure':'cat'},inplace=True)
comb.head()


# In[ ]:


comb['lgb'] = lgb.time_to_failure
comb.head()


# In[ ]:


comb['xgb'] = xgb.time_to_failure
comb.head()


# In[ ]:


comb['time_to_failure'] = comb.mean(numeric_only=True, axis=1)
comb.head()


# In[ ]:


sub = comb.loc[:,['seg_id','time_to_failure']]
sub.head()


# In[ ]:


print(sub.shape)


# In[ ]:


sub.to_csv('submission.csv',index=False)

