#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# # EDA

# In[ ]:


# Reading the train and test file
train_file = pd.read_csv('/kaggle/input/Train.csv')


# In[ ]:


train_file.info()


# **In this dataset, there are total 24 features out of which seven are categorical and remaining 17 are numerical.**

# In[ ]:


train_file['Gender'].value_counts() / len(train_file) * 100


# **Almost 59% of the employees in the dataset are female.**

# ### Identifying Null values

# In[ ]:


null_count = train_file.isnull().sum().reset_index()
null_count.columns = ['feature_name','missing_count']
null_count = null_count[null_count['missing_count'] > 0].sort_values(by='missing_count',ascending=True)
null_value_count = pd.Series(null_count['missing_count'].values, index=null_count['feature_name'])


# In[ ]:


null_value_count / len(train_file) * 100


# **Six features are having null values, out of which VAR2,VAR4,Time_of_Service and Age has the most.**

# ### Correlation Analysis
# 
# I have taken only the numerical columns for calculating the correlation using the pearson method which is highly suitable for continuous data as it takes into account the magnitude of the difference between values i.e. an increase in age from 20 to 21 is same as increase from 60 to 61.Correlation of categorical columns(ordinal) using spearman correlation method.

# In[ ]:


cols = [i for i in train_file.columns if train_file[i].dtype == 'int64' or train_file[i].dtype == 'float64']
corr_num = train_file[cols].corr()
sns.heatmap(corr_num)


# A  very unique heatmap as only the diagonal elements are correlated. It is very well clear from the correlation dataframe that **Age and Time of service** are extremely correlated features hence we can drop one of them in order to cater the multicollinearity assumption of regression. In the current scenario, I will drop **Age** as time of service looks a better option when it comes to predicting employee attrition rate.

# In[ ]:


def encode(data):
    cat_cols = []
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))
            cat_cols.append(col)
    return data , cat_cols


# In[ ]:


train_data, cat_cols = encode(train_file)


# In[ ]:


corr_obj = train_data[cat_cols].corr(method='spearman')
sns.heatmap(corr_obj)


# From the correlation analysis, it is evident that the pairwise correlation of columns is not significant enough to make any inferences.

# ### Distribution of the Target variable i.e. Attrition Rate

# In[ ]:


sns.distplot(np.log1p(train_file['Attrition_rate']))


# **From the plot, it is pretty evident that the distribution is right skewed i.e. positive skew.**

# ### Highlighting the Categorical columns

# In[ ]:


def highlight_cols(s):
    color = '#ADD8E6'
    return 'background-color: %s' % color


# In[ ]:


train_file.style.applymap(highlight_cols, subset=cat_cols)


# In[ ]:





# In[ ]:




