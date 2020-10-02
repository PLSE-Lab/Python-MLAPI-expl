#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore') 


# In[ ]:


# importing helpful analytics libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
# import pandas_profiling as pf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
os.listdir("../input")


# In[ ]:


# import train data
df = pd.read_csv('../input/train.csv', index_col='Id')
df.head()


# In[ ]:


# calculate correlation matrix
corr = df.corr()


# In[ ]:


# filter high (above .35) correlated features with SalePrice
highly_corr = 0.35
filt1 = (corr.SalePrice > highly_corr)
filt2 = (corr.SalePrice < -highly_corr)
filt = filt1 | filt2
corr_filt = corr.loc[filt, 'SalePrice']
cols = corr_filt.index.values
for col in cols:
    print(col, end=' | ')


# In[ ]:


# display highly correlated features
dfc = df[cols].drop_duplicates()
dfc.head()


# In[ ]:


# display basic statistics
dfc.describe().T


# In[ ]:


# desc_pf = pf.ProfileReport(dfc)
# desc_pf


# In[ ]:


dfc.columns


# In[ ]:


# 'LotFrontage' and 'MasVnrArea' has the most number of missing values
cols = ['LotFrontage', 'MasVnrArea']
dfc.loc[:, cols].isna().sum()


# In[ ]:


# calculate mean for LotFrontage
LotFtge_mean = dfc.LotFrontage.mean()
# calculate mean for MasVnrArea
MasVnrAr_mean = dfc.MasVnrArea.mean()
# filling na's with mean
dfc.LotFrontage.fillna(value=LotFtge_mean, inplace=True)
dfc.MasVnrArea.fillna(value=MasVnrAr_mean, inplace=True)
# double-check filling na values with mean value
dfc.loc[:, cols].isna().sum()


# In[ ]:


dfc.describe().T


# In[ ]:


dfc.drop(columns='GarageYrBlt', axis=1, inplace=True)


# In[ ]:


cols = dfc.columns
cols


# In[ ]:


# change data types in dfc
df_types = ['float64', 'int64', 'int64', 'int64', 'float64', 'float64', 'float64', 
            'float64', 'float64', 'int64', 'int64', 'int64', 'int64', 'float64', 'float64']

for i, col in enumerate(cols):
    dfc[cols[i]] = dfc[cols[i]].astype(df_types[i])
# display percentiles
dfc.describe(percentiles=[0.1, 0.15, 0.25, 0.5, 0.75, 0.9]).T


# In[ ]:


# We are going to sort columns in order of higher to lower variability
dfc_var = dfc.var(axis=0)
dff_var = dfc_var.reset_index()
dff_var.columns = ['feature', 'variance']
dff_var.sort_values(by='variance', ascending=False, inplace=True)
dff_var[1:]


# In[ ]:


dfc.head()


# In[ ]:


dfc.dtypes


# In[ ]:


# creating matrix for simple model
X_train = dfc.GrLivArea.values
y_train = dfc.SalePrice.values


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()


# In[ ]:


X_train = X_train.reshape((-1, 1))
X_train.shape, y_train.shape


# In[ ]:


model = dtr.fit(X_train, y_train)
model.score(X_train, y_train)


# In[ ]:


cols = cols.insert(0, 'Id')
cols


# In[ ]:


df_test = pd.read_csv('../input/test.csv', usecols=cols[:-1], index_col='Id')
df_test.head()


# In[ ]:


X_test = df_test.GrLivArea.values
X_test = X_test.reshape((-1, 1))
X_test.shape


# In[ ]:


y_hat = model.predict(X_test)
y_hat


# In[ ]:


df_test.index


# In[ ]:


cols_test = ['Id', 'SalePrice']
df_test['SalePrice'] = y_hat
df_test[:-1].head()


# In[ ]:


df_test.to_csv(path_or_buf='submission.csv', columns=['SalePrice'])


# In[ ]:


os.listdir()


# In[ ]:


kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "Starting this competition"

