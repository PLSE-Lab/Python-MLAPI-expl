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



get_ipython().system('pip install catboost')


# In[ ]:



get_ipython().system('ls')


# In[ ]:



import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn import metrics, preprocessing, model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from collections import defaultdict, Counter
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


import string
from imblearn.over_sampling import SMOTE, ADASYN

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_columns = 100
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_df = pd.read_csv('/kaggle/input/wns-wizard-dataset-tunde/train_NA17Sgz/train.csv')


# In[ ]:


train_df


# In[ ]:



train_df.info()


# In[ ]:



train_df.shape


# In[ ]:


train_df.groupby('is_click')['impression_id'].count()


# In[ ]:


train_df['is_click'].value_counts(normalize=True)


# In[ ]:



print(f'Number of samples in train: {train_df.shape[0]}')
print(f'Number of columns in train: {train_df.shape[1]}')
for col in train_df.columns:
    if train_df[col].isnull().any():
        print(col, train_df[col].isnull().sum())


# In[ ]:


train_df.dtypes


# In[ ]:


missing_data = (((train_df.isnull().sum())*100)/len(train_df))
missing_data


# In[ ]:


item_data_df = pd.read_csv('/kaggle/input/wns-wizard-dataset-tunde/train_NA17Sgz/item_data.csv')


# In[ ]:



item_data_df.head()


# In[ ]:


item_data_df.isnull().sum()


# In[ ]:


item_data_df.shape


# In[ ]:


view_log_df = pd.read_csv('/kaggle/input/wns-wizard-dataset-tunde/train_NA17Sgz/view_log.csv')


# In[ ]:


view_log_df.head()


# In[ ]:


view_log_df.isnull().sum()


# In[ ]:


view_log_df.shape


# In[ ]:


item_view_log_df = pd.merge(view_log_df, item_data_df, on='item_id', how='left')


# In[ ]:


item_view_log_df.shape


# In[ ]:


item_view_log_df.drop_duplicates(inplace=True)


# In[ ]:



item_view_log_df.shape


# In[ ]:


item_view_log_df.head()


# In[ ]:


item_view_log_df[item_view_log_df['user_id'] ==0].head()


# In[ ]:


item_view_log_df.dtypes


# In[ ]:


cols = ['device_type']
for col in cols:
    if item_view_log_df[col].dtype==object:
        print(col)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(item_view_log_df[col].values.astype('str')))
        item_view_log_df[col] = lbl.transform(list(item_view_log_df[col].values.astype('str')))

