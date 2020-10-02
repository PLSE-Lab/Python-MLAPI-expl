#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


train.columns


# I assume that SalesPrice mostly will depend on Neighbourhood, OverallCond, YearBuilt, GrLivArea, TotRmsAbvGrd, SaleType and TotalBsmtSF. Let's plot some scatterplots depending on these variables

# In[ ]:


varss = ['YearBuilt', 'GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF']
for var in varss:
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# Seems like 1st graph is exponentially behaving, while the 2nd and 4th are almost linear 

# Let's see for categorical data feautures

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

varss = ['Neighborhood', 'OverallQual', 'SaleType']
for var in varss:
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000);


# Seems like SalePrice very depends on OveralQual

# To be sure, let's plot heatmap

# In[ ]:


k = 10 #number of variables for heatmap
corrmat = train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# This heatmap shows on which variables SalePrice is correlated with.

# Moreover, from here we can see that SalePrice is correlated with YearBuilt as well 

# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #just checking that there's no missing data missing...


# In[ ]:


from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle


# In[ ]:


test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


total1 = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total1', 'Percent'])
missing_data.head(20)


# In[ ]:


test = test.drop((missing_data[missing_data['Total1'] > 1]).index,1)
test = test.drop(test.loc[test['Electrical'].isnull()].index)
test.isnull().sum().max() #just checking that there's no missing data missing...


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


data = pd.concat([train, test], keys=['train', 'test'])
necessary_columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
for column in train.columns:
    if column not in necessary_columns:
        data.drop(column, axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


# TotalBsmtSF  NA in pred. I suppose NA means 0
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)

# GarageCars  NA in pred. I suppose NA means 0
data['GarageCars'] = data['GarageCars'].fillna(0.0)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] 
data.drop(['TotalBsmtSF', '1stFlrSF'], axis=1, inplace=True)


# In[ ]:


train_data = data.loc['train'].select_dtypes(include=[np.number]).values
test_data = data.loc['test'].select_dtypes(include=[np.number]).values


# In[ ]:


train_y = train.pop('SalePrice')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_data, train_y, test_size=0.2, random_state=200)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression

scores = cross_val_score(LinearRegression(), X_train, y_train, cv=10)
scores


# In[ ]:


scores = cross_val_score(LogisticRegression(C=0.001, max_iter=1000), X_train, y_train, cv=10)
scores


# In[ ]:


model_lin = LinearRegression()
model_lin.fit(X_train, y_train)
y_lin = model_lin.predict(X_test)


# In[ ]:


pd.DataFrame({'Id': test.Id[:292], 'SalePrice': y_lin}).to_csv('house_saleprice.csv', index =False)

