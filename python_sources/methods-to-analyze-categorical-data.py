#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sbn

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


test_data = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
train_data = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')


# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# In[ ]:


test_data['target'] = -1
full_data = pd.concat([train_data,test_data],axis = 0)


# In[ ]:


(train_data.shape,test_data.shape)


# In[ ]:


(full_data[full_data['target'] != -1 ].shape, full_data[full_data['target'] == -1].shape)


# In[ ]:


full_data.dtypes


# In[ ]:


columns1 = ['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4',
            'ord_0','ord_1','ord_2','ord_3','ord_4','day','month','target']
fig, ax = plt.subplots(5,4 , figsize=(20,20))
for variable, subplot in zip(columns1, ax.flatten()):
    sbn.countplot(full_data[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(45)


# In[ ]:


ax = full_data.groupby(['target','bin_0']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("bin_0 Vs target")


# In[ ]:


ax = full_data.groupby(['target','bin_1']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("bin_1 Vs target")


# In[ ]:


ax = full_data.groupby(['target','bin_2']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("bin_2 Vs target")


# In[ ]:


ax = full_data.groupby(['target','bin_3']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("bin_3 Vs target")


# In[ ]:


ax = full_data.groupby(['target','bin_4']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("bin_4 Vs target")


# In[ ]:


ax = full_data.groupby(['target','nom_0']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("nom_0 Vs target")


# In[ ]:


ax = full_data.groupby(['target','nom_1']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("nom_1 Vs target")


# In[ ]:


ax = full_data.groupby(['target','nom_2']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("nom_2 Vs target")


# In[ ]:


ax = full_data.groupby(['target','nom_3']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("nom_3 Vs target")


# In[ ]:


ax = full_data.groupby(['target','nom_4']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("nom_4 Vs target")


# In[ ]:


ax = full_data.groupby(['target','ord_0']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("ord_0 Vs target")


# In[ ]:


ax = full_data.groupby(['target','ord_1']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("ord_1 Vs target")


# In[ ]:


ax = full_data.groupby(['target','ord_2']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("ord_2 Vs target")


# In[ ]:


ax = full_data.groupby(['target','ord_3']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("ord_3 Vs target")


# In[ ]:


ax = full_data.groupby(['target','ord_4']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("ord_4 Vs target")


# In[ ]:


ax = full_data.groupby(['target','day']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("day Vs target")


# In[ ]:


ax = full_data.groupby(['target','month']).size().unstack()      .plot(kind='bar', stacked=True, figsize=(18, 7))
ax.set_xlabel("mont Vs target")


# In[ ]:


import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder  
def categorical_ass(dataset, nominal_columns=None, mark_columns=False, theil_u=True, plot=True, return_results=True, 
                 ax=None, **kwargs):
    columns = dataset.columns
      
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if dataset[columns[j]].dtype != int:
                    cat_encoder = LabelEncoder()
                    j_col = cat_encoder.fit_transform(dataset[columns[j]])
                else:
                    j_col = dataset[columns[j]]
                    
                if dataset[columns[i]].dtype != int:
                    cat_encoder = LabelEncoder()
                    i_col = cat_encoder.fit_transform(dataset[columns[i]])
                else:
                    i_col = dataset[columns[i]]
                cell, _ = ss.pearsonr(i_col, j_col)
                corr[columns[i]][columns[j]] = cell
                corr[columns[j]][columns[i]] = cell
                
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        if ax is None:
            plt.figure(figsize=(17,6))
        sbn.heatmap(corr, annot=kwargs.get('annot', True), fmt=kwargs.get('fmt', '.2f'), ax=ax)
        if ax is None:
            plt.show()
    if return_results:
        return corr


# In[ ]:



columns1 = ['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4',
            'ord_0','ord_1','ord_2','ord_3','ord_4','day','month','target']
corr = categorical_ass(dataset = full_data[columns1])
print(corr)


# In[ ]:


full_data = full_data[['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_0','ord_1','ord_2','day','month','target']]
cols = ['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_0','ord_1','ord_2','day','month']
full_data = pd.get_dummies(full_data,prefix = cols,columns = cols)


# In[ ]:


train_x_data = full_data[full_data['target'] != -1].copy()
test_y_data = full_data[full_data['target'] == -1].copy()


# In[ ]:


x_data = train_x_data.drop('target',axis = 1)
y_data = train_x_data[['target']]


# In[ ]:


from sklearn.model_selection import train_test_split ,GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=44)

