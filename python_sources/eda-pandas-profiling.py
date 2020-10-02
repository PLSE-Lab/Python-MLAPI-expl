#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pandas_profiling


# In[ ]:


train = pd.read_csv("/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv")


# In[ ]:


train.shape


# In[ ]:


pandas_profiling.ProfileReport(train)

