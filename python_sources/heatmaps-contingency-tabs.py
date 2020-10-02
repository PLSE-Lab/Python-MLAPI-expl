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


train_transaction_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')


# In[ ]:


import seaborn as sns
sns.set(rc={'figure.figsize':(15,12)})


# In[ ]:


sns.heatmap(pd.crosstab(train_transaction_df.isFraud, train_transaction_df.ProductCD), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(train_transaction_df.isFraud, train_transaction_df.card4), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(train_transaction_df.isFraud, train_transaction_df.card6), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(train_transaction_df.P_emaildomain, train_transaction_df.isFraud), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(train_transaction_df.R_emaildomain, train_transaction_df.isFraud), annot = True, fmt = "d")


# In[ ]:


sns.heatmap(pd.crosstab(train_transaction_df.M1, train_transaction_df.isFraud), annot = True, fmt = "d")


# In[ ]:


drp_row = train_transaction_df.dropna()


# In[ ]:


drp_row.shape


# In[ ]:


drp_col = train_transaction_df.dropna(axis = 'columns')


# In[ ]:


drp_col.shape


# In[ ]:


drp_col.head()


# In[ ]:


drp_col.nunique()


# In[ ]:


crstb = pd.crosstab(drp_col.card1, drp_col.isFraud)


# In[ ]:


crstb


# In[ ]:


crstb_two = crstb.unstack().reset_index().rename(columns={0:"cnt"})


# In[ ]:


crstb_two


# In[ ]:


crstb_two.sort_values(by=['isFraud', 'cnt'], ascending=False)


# In[ ]:


import seaborn as sns


# In[ ]:


sns.regplot(x = crstb_two.card1, y = crstb_two.isFraud, logistic=True)

