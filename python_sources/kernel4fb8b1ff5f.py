#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


item_sales = pd.read_csv('../input/Train_UWu5bXk.csv')
item_sales


# In[ ]:


item_sales.isna().sum()/item_sales.count()*100


# In[ ]:


item_sales = item_sales.drop('Outlet_Size', axis=1)
item_sales


# In[ ]:


item_sales['Item_Weight'] = item_sales['Item_Weight'].fillna(item_sales['Item_Weight'].mean())


# In[ ]:


item_sales['Item_Weight'].isna().sum()


# In[ ]:


item_sales.var()


# In[ ]:


item_sales.corr()


# In[ ]:


item_sales_reduced = item_sales[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 'Item_Outlet_Sales']]
item_sales_reduced


# In[ ]:


from sklearn.preprocessing import StandardScaler
wieght_transformer = StandardScaler()
weights = wieght_transformer.fit_transform(item_sales_reduced.values)
weights


# In[ ]:


Y = weights[:, -1]
X = weights[:, :-1]


# In[ ]:


Y.shape


# In[ ]:


X.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X, Y)


# In[ ]:


lr_model.coef_


# In[ ]:


# BFE will be HW


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


from sklearn.datasets import load_iris
dataset = load_iris()
X_iris = dataset['data']
Y_iris = dataset['target']


# In[ ]:


dataset.keys()


# In[ ]:


pca_model = PCA()
pca_model.fit(X)


# In[ ]:


pca_model.components_


# In[ ]:


pca_model.explained_variance_


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.pairplot(item_sales_reduced)


# In[ ]:


pca_model_iris = PCA(n_components=2)
pca_model_iris.fit(X_iris)


# In[ ]:


pca_model_iris.explained_variance_


# In[ ]:


pca_model_iris.components_


# In[ ]:


sns.pairplot(pd.DataFrame(X_iris, columns=dataset['feature_names']))


# In[ ]:




