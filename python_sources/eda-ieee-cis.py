#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno


# In[ ]:


train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')


# In[ ]:


train_identity.describe()


# In[ ]:


train_identity.tail()


# In[ ]:


train_identity.shape , train_transaction .shape ,test_identity.shape , test_transaction .shape


# In[ ]:


numeric_features = train_identity.select_dtypes(include=[np.number])

numeric_features.columns


# In[ ]:


numeric_features = train_transaction.select_dtypes(include=[np.number])

numeric_features.columns


# In[ ]:


categorical_features = train_identity.select_dtypes(include=[np.object])
categorical_features.columns


# In[ ]:


msno.matrix(train_identity.sample(250))


# In[ ]:


msno.heatmap(train_identity)


# In[ ]:


msno.bar(train_identity.sample(1000))


# In[ ]:


msno.dendrogram(train_identity)


# In[ ]:


train_identity.skew(), train_identity.kurt()

