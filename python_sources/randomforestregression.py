#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


def editMissVals(df):
    dfna = (df.isnull().sum() / len(df)) * 100
    dfna = dfna.drop(dfna[dfna == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :dfna})
    print(missing_data)
    
    df.PoolQC = df.PoolQC.fillna("None")
    df.MiscFeature = df.MiscFeature.fillna("None")
    df.Alley = df.Alley.fillna("None")
    df.Fence = df.Fence.fillna("None")
    df.FireplaceQu = df.FireplaceQu.fillna("None")
    df.LotFrontage = df.groupby("Neighborhood").LotFrontage.transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        df[col] = df[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        df[col] = df[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')
    df.MasVnrType = df.MasVnrType.fillna("None")
    df.MasVnrArea = df.MasVnrArea.fillna(0)
    df.MSZoning = df.MSZoning.fillna(df.MSZoning.mode()[0])
    df = df.drop(['Utilities'], axis=1)
    df.Functional = df.Functional.fillna("Typ")
    df.Electrical = df.Electrical.fillna(df.Electrical.mode()[0])
    df.KitchenQual = df.KitchenQual.fillna(df.KitchenQual.mode()[0])
    df.Exterior1st = df.Exterior1st.fillna(df.Exterior1st.mode()[0])
    df.Exterior2nd = df.Exterior2nd.fillna(df.Exterior2nd.mode()[0])
    df.SaleType = df.SaleType.fillna(df.SaleType.mode()[0])
    df.MSSubClass = df.MSSubClass.fillna("None")
    return df


# In[ ]:


df = editMissVals(df)
df.head(10)


# In[ ]:


df = pd.get_dummies(df)
df.head(10)


# In[ ]:


X = df.drop('SalePrice', axis=1)
y = df.SalePrice


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(n_estimators=200)


# In[ ]:


rf.fit(X, y)


# In[ ]:


scores = rf.score(X_test, y_test)
print("accuracies     = ", scores)
print("mean accuracy = %4.2f" % (scores.mean()))b


# In[ ]:


predicted = rf.predict(X_test)


# In[ ]:


resultDf = pd.DataFrame(columns=['target', 'predicted'])


# In[ ]:


resultDf.target = y_test
resultDf.predicted = predicted
resultDf


# In[68]:


from sklearn.metrics import explained_variance_score
explained_variance_score(resultDf.target, resultDf.predicted)


# In[69]:


from sklearn.metrics import r2_score
r2_score(resultDf.target, resultDf.predicted)


# In[70]:


from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(resultDf.target, resultDf.predicted)


# In[ ]:


resultDf.corr()


# In[ ]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
train = editMissVals(train)
test = editMissVals(test)
train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[ ]:


train_X = train.drop('SalePrice', axis=1)
train_y = train.SalePrice


# In[ ]:


train_columns = train_X.columns
test_columns = test.columns

filt1 = [col not in test_columns for col in train_columns]
filt2 = [col not in train_columns for col in test_columns]
unmatches1 = train_columns[filt1]
unmatches2 = test_columns[filt2]

train_X = train_X.drop(unmatches1, axis=1)
test = test.drop(unmatches2, axis=1)


# In[ ]:


print(len(train_X.columns), len(test.columns))


# In[ ]:


rf = RandomForestRegressor(n_estimators=200)
rf.fit(train_X, train_y)


# In[ ]:


test_predicted = rf.predict(test)


# In[ ]:


test_resultDf = pd.DataFrame(columns=['Id', 'SalePrice'])
test_resultDf.Id = test.index
test_resultDf.SalePrice = test_predicted
test_resultDf


# In[ ]:


test_resultDf.to_csv('sample_submission.csv', index=None)


# In[ ]:


test_resultDf

