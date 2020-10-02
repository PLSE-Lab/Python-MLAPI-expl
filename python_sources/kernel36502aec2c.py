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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_trains = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df_trains.columns


# In[ ]:


df_trains['SalePrice'].describe()


# In[ ]:


sns.distplot(df_trains['SalePrice']);


# In[ ]:


print("Skewness: %f" % df_trains['SalePrice'].skew())
print("Kurtosis: %f" % df_trains['SalePrice'].kurt())


# In[ ]:


var = 'GrLivArea'
data = pd.concat([df_trains['SalePrice'], df_trains[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


var = 'TotalBsmtSF'
data = pd.concat([df_trains['SalePrice'], df_trains[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


var = 'OverallQual'
data = pd.concat([df_trains['SalePrice'], df_trains[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


var = 'YearBuilt'
data = pd.concat([df_trains['SalePrice'], df_trains[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# In[ ]:


corrmat = df_trains.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_trains[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_trains[cols], size = 2.5)
plt.show();


# In[ ]:


total = df_trains.isnull().sum().sort_values(ascending=False)
percent = (df_trains.isnull().sum()/df_trains.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


df_trains = df_trains.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_trains = df_trains.drop(df_trains.loc[df_trains['Electrical'].isnull()].index)
df_trains.isnull().sum().max() 


# In[ ]:


saleprice_scaled = StandardScaler().fit_transform(df_trains['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[ ]:


var = 'GrLivArea'
data = pd.concat([df_trains['SalePrice'], df_trains[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


df_trains.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_trains = df_trains.drop(df_trains[df_trains['Id'] == 1299].index)
df_trains = df_trains.drop(df_trains[df_trains['Id'] == 524].index)


# In[ ]:


var = 'TotalBsmtSF'
data = pd.concat([df_trains['SalePrice'], df_trains[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


sns.distplot(df_trains['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_trains['SalePrice'], plot=plt)


# In[ ]:


df_trains['SalePrice'] = np.log(df_trains['SalePrice'])


# In[ ]:


sns.distplot(df_trains['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_trains['SalePrice'], plot=plt)


# In[ ]:


sns.distplot(df_trains['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_trains['GrLivArea'], plot=plt)


# In[ ]:


df_trains['GrLivArea'] = np.log(df_trains['GrLivArea'])


# In[ ]:


sns.distplot(df_trains['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_trains['GrLivArea'], plot=plt)


# In[ ]:


sns.distplot(df_trains['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_trains['TotalBsmtSF'], plot=plt)


# In[ ]:


df_trains['HasBsmt'] = pd.Series(len(df_trains['TotalBsmtSF']), index=df_trains.index)
df_trains['HasBsmt'] = 0 
df_trains.loc[df_trains['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[ ]:


df_trains.loc[df_trains['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_trains['TotalBsmtSF'])


# In[ ]:


sns.distplot(df_trains[df_trains['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_trains[df_trains['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[ ]:


plt.scatter(df_trains['GrLivArea'], df_trains['SalePrice']);


# In[ ]:


plt.scatter(df_trains[df_trains['TotalBsmtSF']>0]['TotalBsmtSF'], df_trains[df_trains['TotalBsmtSF']>0]['SalePrice']);


# In[ ]:


df_trains = pd.get_dummies(df_trains)


# In[ ]:




