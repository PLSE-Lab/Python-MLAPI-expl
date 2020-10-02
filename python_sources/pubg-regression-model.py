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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train_V2.csv", index_col='Id')
print(train.shape)
train.head()
train.describe()


# In[ ]:


train.isna().sum()


# In[ ]:


filt = train['winPlacePerc'].isna()
train[filt]


# In[ ]:


train = train.fillna(0) 


# In[ ]:


train.corr().style.format("{:.2%}").highlight_min()
correlations = train.corr()
sns.heatmap(correlations)

def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
plot_correlation_heatmap(train)


# In[ ]:


X = train['walkDistance'].values.reshape(-1,1)


# In[ ]:


X[:10]


# In[ ]:


y = train['winPlacePerc'].values
y[:10]


# In[ ]:


## Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cvs_lr = cross_val_score(lr, X, y, cv=15)
cvs_lr.mean(), cvs_lr.std()


# In[ ]:


## Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()


# In[ ]:


cvs_dtr = cross_val_score(dtr, X, y)
cvs_dtr.mean(), cvs_dtr.std()


# In[ ]:





# In[ ]:


dtr.fit(X,y)


# In[ ]:


test = pd.read_csv("../input/test_V2.csv", index_col='Id')
test.head()


# In[ ]:


test.isna().sum().sum()


# In[ ]:


X_test = test['walkDistance'].values.reshape(-1,1)
X_test[:10]


# In[ ]:


predictions = dtr.predict(X_test).reshape(-1,1)


# In[ ]:


dfpredictions = pd.DataFrame(predictions, index=test.index).rename(columns={0:'winPlacePerc'})
dfpredictions.head(15)


# In[ ]:


dfpredictions.to_csv('submission.csv', header=True)


# In[ ]:




