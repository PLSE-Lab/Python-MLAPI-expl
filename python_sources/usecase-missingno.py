#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('ls', '../input')


# In[ ]:


import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


target_col = "SalePrice"
target = train[target_col]
train.drop(columns=[target_col], inplace=True)


# In[ ]:


mix = pd.concat([train, test])


# # basic_information

# In[ ]:


print("train.shape: {}".format(train.shape))
print("test.shape: {}".format(test.shape))
print("mix.shape: {}".format(mix.shape))


# In[ ]:


train.head()


# In[ ]:


test.head()


# # missingno_matrix

# In[ ]:


msno.matrix(train, labels=list(train.columns))


# In[ ]:


msno.matrix(test, labels=list(test.columns))


# In[ ]:


msno.matrix(mix, labels=list(mix.columns))


# # missingno_bar

# In[ ]:


msno.bar(train)


# In[ ]:


msno.bar(test)


# In[ ]:


msno.bar(mix)


# # missingno_heatmap

# In[ ]:


msno.heatmap(train)


# In[ ]:


msno.heatmap(test)


# In[ ]:


msno.heatmap(mix)

