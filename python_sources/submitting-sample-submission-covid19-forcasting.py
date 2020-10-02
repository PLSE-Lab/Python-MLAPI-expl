#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)

