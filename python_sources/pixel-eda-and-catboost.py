#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[ ]:


def load_train_data():
    return pd.read_csv('../input/quickdraw_train.csv', index_col='id')

def load_test_data():
    return pd.read_csv('../input/quickdraw_test_x.csv', index_col='id')


# In[ ]:


df_train = load_train_data()
df_test = load_test_data()


# In[ ]:


def make_pixel_agg(df):
    columns = [f for f in df.columns if 'pix' in f]
    df['mean'] = df[columns].mean(axis = 1)
    df['var'] = df[columns].var(axis = 1)
    df['no_dead'] = df[columns][df[columns] == 0].count(axis=1)
    return df

def pixel_location(df):
    columns = [f for f in df.columns if 'pix' in f]
    df['up_or_down'] = np.where(df[columns[:392]].sum(axis=1) > df[columns[392:]].sum(axis=1), 1, 0)
    df['biggest_pixel'] = np.argmax(df[columns].values, axis=1)
    return df


# In[ ]:


df_train = make_pixel_agg(df_train)
df_train = pixel_location(df_train)

df_test = make_pixel_agg(df_test)
df_test = pixel_location(df_test)


# In[ ]:


df_train.head()


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(15, 15)})
ax = sns.barplot(x="subcategory", y="mean", data=df_train)
plt.setp(ax.get_xticklabels(), rotation=90)


# In[ ]:


ax = sns.barplot(x="subcategory", y="var", data=df_train)
plt.setp(ax.get_xticklabels(), rotation=90)


# In[ ]:


ax = sns.barplot(x="subcategory", y="no_dead", data=df_train)
plt.setp(ax.get_xticklabels(), rotation=90)


# ## **Catboost!**

# In[ ]:


import catboost as cb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[ ]:


df_train.columns


# In[ ]:


columns_to_use = ['mean', 'var','no_dead', 'up_or_down', 'biggest_pixel']

X_train, X_test, y_train, y_test = train_test_split(df_train[columns_to_use],
                                                   df_train['category'].values,
                                                   stratify = df_train['category'])
le = LabelEncoder()
y_train = le.fit_transform(y_train.flatten())
y_test  = le.transform(y_test.flatten())


# In[ ]:


X_train.columns


# In[ ]:


clf = cb.CatBoostClassifier(iterations = 500,learning_rate=0.1, eval_metric = 'Accuracy')
clf.fit(X_train, y_train, cat_features = [3], eval_set = [(X_train, y_train),(X_test, y_test)])


# In[ ]:


clf.feature_importances_


# 72% Accuracy for Binary Classification - Food or Animal?

# In[ ]:




