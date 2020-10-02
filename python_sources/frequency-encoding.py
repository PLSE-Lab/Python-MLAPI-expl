#!/usr/bin/env python
# coding: utf-8

# # Frequency Encoding

# **Frequency Encoding is an encoding technique which encodes categorical feature values to their frequencies**
#     
# ***This will preserve the information about the values of distributions.***

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat/train.csv')
test = pd.read_csv('../input/cat-in-the-dat/test.csv')

train.shape,test.shape


# In[ ]:


train.head()


# **Let's find the Frequency of each unique categorical values of feature "nom_1"**

# In[ ]:


enc_nom_1 = (train.groupby('nom_1').size()) / len(train)
enc_nom_1


# In[ ]:


train['nom_1_encode'] = train['nom_1'].apply(lambda x : enc_nom_1[x])


# In[ ]:


train.head()


# In[ ]:




