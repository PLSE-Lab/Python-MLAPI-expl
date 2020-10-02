#!/usr/bin/env python
# coding: utf-8

# ## Grid visualizing feature correlations

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


sns.pairplot(train[[col for col in train.columns if 'cont' in col]])

