#!/usr/bin/env python
# coding: utf-8

# # Simple data cleaning, Canada Covid-19
# ### Jupyter notebook author: Tao Shan
# 1. [prepare data](#1) 
# 2. [explore data](#2)
# 3. [null value and data engineering](#3)
# 4. [predicting model and submit solution](#4)

# <a id="1"></a>
# 1.prepare data

# In[ ]:


#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# In[ ]:


dt_train = pd.read_csv('../input/coronaviruscovid19-canada/cases.csv')
print('Training data shape is {}'.format(dt_train.shape))


# In[ ]:


dt_train.head(3)


# <a id="2"></a>
# 2.explore data

# Since some of the values (not reported) is more than 90%, should be deleted 

# In[ ]:


drop_col = []
for col in dt_train.columns:
    if dt_train[col].value_counts().iloc[0] / len(dt_train[col].dropna()) > 0.9:
        drop_col.append(col)
drop_col


# In[ ]:


#number of unique categories
for col_name in dt_train.columns:
    print(col_name, dt_train[col_name].nunique())


# Since case_id , provincial_case_id and case_source are useless, drop them.

# In[ ]:


dt_train.drop(['case_id', 'provincial_case_id', 'case_source'], axis = 1, inplace = True)
dt_train.head(3)


# Since they are all categorical variable, draw raw graphs

# In[ ]:


dt_columns = dt_train.columns.tolist()
fig = plt.figure(figsize=(32,32))
for index,col in enumerate(dt_columns):
    plt.subplot(4,4,index+1)
    sns.countplot(x = dt_train[dt_columns].iloc[:,index], data = dt_train[dt_columns])
fig.tight_layout(pad=1.0)


# In[ ]:


#find null values percentages
null_percentage = dt_train.isnull().sum().sort_values(ascending = False)/dt_train.shape[0]
print(null_percentage)


# In[ ]:


del_list = []
fill_list = []
train_columns = dt_train.columns.tolist()
for col in train_columns:
    if null_percentage[col] >0.75:
        del_list.append(col)
    else:
        fill_list.append(col)
del_list


# In[ ]:


dt_train.drop(del_list, axis = 1, inplace = True)
dt_train.sample(3)


# <a id="3"></a>
# 3. null value and data engineering

# In[ ]:


#simplify this step, fill with a value
dt_train['method_note'] = dt_train['method_note'].fillna(10000)
dt_train.isnull().sum().sort_values(ascending = False)


# In[ ]:


#label incoder
dt_train= dt_train.apply(lambda series: pd.Series(
    LabelEncoder().fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index))
dt_train.head(3)


# In[ ]:


dt_train.to_csv('../../kaggle/working/Try_dataset_aftercleanning.csv', header=True, index=False)


# In[ ]:




