#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Getting a glimpse of the data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print("The shape of the training data is",train.shape)
train.head()


# In[ ]:


print("The shape of the test data is",test.shape)
test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# - The train data consists of 9557 entries and 143 columns.
# - 5 features are of datatype object.
# - Since the memory usage is pretty low we don't need to worry about the size of the datatype.
# 
# The distribution of the data types is similar in test data. Just the size of test data is more than the train data in terms of the number of entries. 

# ## Missing data

# In[ ]:


def missing_data(data): #calculates missing values in each column
    total = data.isnull().sum().reset_index()
    total.columns  = ['Feature_Name','Missing_value']
    total_val = total[total['Missing_value']>0]
    total_val = total.sort_values(by ='Missing_value',ascending=False)
    return total_val


# In[ ]:


missing_data(train).head()


# In[ ]:


missing_data(test).head()


# Five features have missing value in the training set out of which two have just 5 missing values which is fine but there are three features having more than 71 % null values. The features having maximum null values are :
# - Years behind in school
# - Number of tablets household owns
# - Monthly rent payment
# 
# The test set also has null values in the same columns as train set.

# In[ ]:


train.update(train[['rez_esc','v18q1','v2a1']].fillna(0))
test.update(test[['rez_esc','v18q1','v2a1']].fillna(0))

train['meaneduc'].fillna((train['meaneduc'].median()), inplace=True)
test['meaneduc'].fillna((test['meaneduc'].median()), inplace=True)


# ## Target Distribution

# In[ ]:


train['Target'].value_counts()


# In[ ]:


vals = train['Target']
labels = 'extreme poverty','moderate poverty','vulnerable households','non vulnerable households'
values = [sum(vals == 1), sum(vals == 2),sum(vals == 3),sum(vals == 4)]

Plot = go.Pie(labels=labels, values=values,marker=dict(line=dict(color='#fff', width= 3)))
layout = go.Layout(title='target distribution', height=400)
fig = go.Figure(data=[Plot], layout=layout)
iplot(fig)


#   ## Relationship between the number of rooms and target

# In[ ]:


room_color_table = pd.crosstab(index=train["rooms"],columns=train["Target"])
room_color_table.plot(kind="bar",figsize=(11,11),stacked=True)


# ## Relationship between number of rooms and total number of people in the household

# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
sns.pointplot(x=train['rooms'],y=train['r4t3'], color='Black')
plt.xlabel('Number of rooms')
plt.ylabel('Total number of people in the house')
plt.show()


# ## Converting datatypes - object to float

# In[ ]:


mapping = {"yes": 1, "no": 0}

columns = [cols for cols in train.columns if train[cols].dtype == 'object' and cols not in ['Id','idhogar']]
train[columns] = train[columns].replace(mapping).astype(np.float64)


# ## Dropping squared & unwanted features

# In[ ]:


# Preparing train data
cols = [c for c in train.columns if c.startswith('SQB')] + ['Id','idhogar','agesq']
X = train.drop(cols,axis=1)

# Preparing test data
col = [x for x in test.columns if x.startswith('SQB')] + ['Id','idhogar','agesq']
X_test = test.drop(col, axis=1)
X_test = X_test.dropna(axis=0)


# ## Correlation of features with the target 

# In[ ]:


corrmat = X.corr().abs()['Target'].sort_values(ascending=False).drop('Target')
corr_df = corrmat.to_frame(name='values')
corr_df.index[corr_df['values']<0]


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(corr_df[:40])


# In[ ]:


corr_df[corr_df['values']>0.3] # Number of features having high correlation with the target


# #### Please upvote the kernel if you liked it and if you have any suggestions or corrections to make do comment below. I would love to hear from you guys.

# In[ ]:




