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


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')
sns.set(font_scale=1.5)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train =pd.read_csv("../input/train.csv")
test =pd.read_csv('../input/test.csv')


# In[ ]:


corr = train[train.SalePrice>1].corr()
top_corr_cols = corr[abs((corr.SalePrice)>=.6)].SalePrice.sort_values(ascending=False).keys()
top_corr_cols
top_corr = corr.loc[top_corr_cols, top_corr_cols]
dropSelf = np.zeros_like(top_corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
plt.figure(figsize=(15, 10))
sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)
sns.set(font_scale=0.5)
plt.show()
del corr, dropSelf, top_corr


# In[ ]:


var = 'GrLivArea'
G_L = pd.concat([train['SalePrice'], train[var]], axis=1)
train.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#box plot overallqual/saleprice
var = 'SaleCondition'
Quall = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=train)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


import scipy.stats as st
plt.figure(1); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.johnsonsu);


# In[ ]:


y = train['SalePrice']


# In[ ]:


train_Id = train.iloc[:,train.columns == 'Id']
train = train.iloc[:,train.columns != 'Id']


# In[ ]:


test_Id = test.Id
test = test.iloc[:,test.columns != 'Id']


# In[ ]:


train = train.drop("SalePrice", axis=1)


# In[ ]:


train.shape, test.shape


# In[ ]:


train.drop("MSZoning", axis=1, inplace=True)


# In[ ]:


test.drop("MSZoning", axis=1, inplace=True)


# In[ ]:


train.shape, test.shape


# In[ ]:


new =pd.concat([train, test])


# In[ ]:


for i in ['MSSubClass', 'OverallQual', 'OverallCond']:
    new[i] = new[i].astype('category')


# In[ ]:


new['Alley'].value_counts()


# In[ ]:


new["Alley"].fillna("NA", inplace = True)


# In[ ]:


new["MiscFeature"].fillna("NA", inplace = True)


# In[ ]:


new["PoolQC"].fillna("NA", inplace = True)


# In[ ]:


new["Fence"].fillna("NA", inplace = True)


# In[ ]:


new["FireplaceQu"].fillna(method='ffill', inplace=True)


# In[ ]:


new["FireplaceQu"].fillna("NA", inplace = True)


# In[ ]:


new["GarageType"].fillna("NA", inplace = True)


# In[ ]:


new["GarageFinish"].fillna("NA", inplace = True)   


# In[ ]:


new["GarageQual"].fillna("NA", inplace = True)   


# In[ ]:


new["GarageCond"].fillna("NA", inplace = True)   


# In[ ]:


new["GarageYrBlt"].fillna(method='ffill', inplace=True)


# In[ ]:


new["MasVnrType"].fillna(method='ffill', inplace=True)


# In[ ]:


new['MasVnrArea'].fillna(method='ffill', inplace=True)


# In[ ]:


new['BsmtQual'].fillna('NA', inplace=True)


# In[ ]:


new["BsmtCond"].fillna("NA", inplace = True)


# In[ ]:


new["BsmtExposure"].fillna("NA", inplace = True)


# In[ ]:


new["BsmtFinType1"].fillna("NA", inplace = True)


# In[ ]:


new["BsmtFinType2"].fillna("NA", inplace = True) 


# In[ ]:


new["Electrical"].isnull().sum()


# In[ ]:


new["Electrical"].fillna(method='ffill', inplace=True)


# In[ ]:


missing = new.isnull().sum()
missing = missing[missing > 0]
missing


# In[ ]:


new["Exterior1st"].fillna(method='ffill', inplace=True)


# In[ ]:


new["Exterior2nd"].fillna(method='ffill', inplace=True)


# In[ ]:


new["BsmtFinSF1"].fillna(method='ffill', inplace=True)


# In[ ]:


new["BsmtFinSF2"].fillna(method='ffill', inplace=True)


# In[ ]:


new["TotalBsmtSF"].fillna(method='ffill', inplace=True)


# In[ ]:


new["GarageCars"].fillna("NA", inplace = True)


# In[ ]:


new["BsmtUnfSF"].fillna(method='ffill', inplace=True)
new["BsmtFullBath"].fillna(method='ffill', inplace=True)
new["BsmtHalfBath"].fillna(method='ffill', inplace=True)
new["KitchenQual"].fillna(method='ffill', inplace=True)
new["Functional"].fillna(method='ffill', inplace=True)
new["GarageArea"].fillna(method='ffill', inplace=True)
new["SaleType"].fillna(method='ffill', inplace=True)
new["LotFrontage"].fillna(method='ffill', inplace=True)
new["Utilities"].fillna(method='ffill', inplace=True)


# In[ ]:


missing = new.isnull().sum()
missing = missing[missing > 0]
missing


# In[ ]:


dummies_test = pd.get_dummies(new, drop_first=True)


# In[ ]:


dummies_test.shape


# In[ ]:


X =dummies_test.iloc[:1460, :]


# In[ ]:


test_pred =  dummies_test.iloc[1460:, :]


# In[ ]:


test_pred.shape


# In[ ]:


X.shape , y.shape


# In[ ]:


test_pred.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


# In[ ]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:


model.score(X_test, y_test), model.score(X_train, y_train)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV 

cross_val_score(Lasso(), X_train , y_train, cv=5).mean()


# In[ ]:


Ridge = Ridge(alpha=1,normalize=False )

# Fit the regressor to the data
Ridge.fit(X_train,y_train)
Ridge_pred= Ridge.predict(X_test)
Ridge.score(X_test, y_test)


# In[ ]:


predictions = Ridge.predict(test_pred)


# In[ ]:


submit = pd.DataFrame()
submit['Id'] = test_Id
submit['SalePrice'] = predictions
submit.to_csv("Ridgelast_pred.csv", index=False)

