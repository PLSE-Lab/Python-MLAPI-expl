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

# Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 100
pd.set_option('display.max_columns', 100)


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')
df = pd.concat([train_df,test_df], ignore_index= True)
df.head()


# In[ ]:


width = []
width.append(df['LotArea']/df['LotFrontage'])
value = np.nanmean(width)
print(value)

clean_df = df

clean_df['LotFrontage'] = df['LotFrontage'].fillna(df['LotArea']/value)


# In[ ]:


obj_col = ['MSZoning','Exterior1st','Exterior2nd','Electrical','KitchenQual','Alley', 'Street', 'BldgType','Utilities','MasVnrType','SaleType','Functional','HouseStyle','Heating']
num_col = ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea', 'GarageCars', 'GarageArea']

for col in obj_col:
    clean_df[col] = df[col].fillna(df[col].mode()[0])
    
for col in num_col:
    clean_df[col] = df[col].fillna(0)


# In[ ]:


# clean_df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
clean_df = clean_df.drop('TotalBsmtSF', axis=1)
clean_df = clean_df.drop('1stFlrSF', axis=1)
clean_df = clean_df.drop('2ndFlrSF', axis=1)


# In[ ]:


clean_df.isnull().sum()


# In[ ]:


# clean_df['BsmtQual'] = df['BsmtQual'].fillna(0)
# clean_df['BsmtQual'] = df.replace({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

cols = ['BsmtCond','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC','Fence']

for col in cols:
    clean_df[col] = df[col].fillna(0)
    clean_df[col] = clean_df[col].replace({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
    

clean_df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
clean_df['BsmtFinType1'] = df['BsmtFinType1'].fillna('No')
clean_df['BsmtFinType2'] = df['BsmtFinType2'].fillna('No')
clean_df['GarageType'] = df['GarageType'].fillna('No')
clean_df['GarageYrBlt'] = df['GarageYrBlt'].fillna('No')
clean_df['GarageFinish'] = df['GarageFinish'].fillna('No')
clean_df['MiscFeature'] = df['MiscFeature'].fillna('Othr')
# clean_df['BsmtExposure'] = clean_df['BsmtExposure'].replace()


print(clean_df['GarageType'].unique())
print(df['BsmtQual'].unique())


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# clean_df[col] = le.fit_transform(clean_df['GarageYrBlt'])
le_cols = obj_col.extend(cols)
for col in cols:
    clean_df[col] = le.fit_transform(clean_df[col].astype('str'))
for col in obj_col:
    clean_df[col] = le.fit_transform(clean_df[col].astype('str'))
for col in ['BsmtExposure','BsmtFinType1','BsmtFinType2','MiscFeature','GarageType','GarageYrBlt','GarageFinish','PavedDrive','CentralAir','Condition1','Condition2','ExterCond','ExterQual','Foundation','SaleCondition','RoofStyle','RoofMatl','Neighborhood','LotShape','LotConfig','LandSlope','LandContour']:
    clean_df[col] = le.fit_transform(clean_df[col].astype('str')) 
# # print(clean_df.dtypes)


# In[ ]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

clean_df = clean_dataset(clean_df)
clean_df


# In[ ]:


test_df = clean_df[clean_df['SalePrice'] == 0]
train_df = clean_df[clean_df['SalePrice'] != 0]


# In[ ]:


from sklearn.model_selection import train_test_split

X = train_df.drop('SalePrice', axis=1)
Y = train_df['SalePrice']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,train_size=0.75, test_size=0.25)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test)

print('Training shape: ', X_train.shape, Y_train.shape)
print('Valid shape: ', X_valid.shape, Y_valid.shape)
print('Test shape: ', X_test.shape, Y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression, ElasticNet, Lasso,  BayesianRidge, Ridge, SGDRegressor, LogisticRegression
from sklearn.metrics import accuracy_score
# regression = Ridge()
# regression.fit(X_train, Y_train)
# # y_pred_train = regression.predict(X_train)
# y_pred_test = regression.predict(X_test)
# # y_pred_valid = regression.predict(X_valid)
# print(y_pred_test)
# print(np.round(np.sqrt(metrics.mean_squared_error(np.log1p(X_train),np.log1p(y_pred_train))),4))
ridge = Ridge()
ridge.fit(X_train,Y_train)
y_prob = ridge.predict(X_train)
y_pred = np.asarray([np.argmax(line) for line in y_prob])
yp_test = ridge.predict(X_test)
test_preds = np.asarray([np.argmax(line) for line in yp_test])
print(accuracy_score(Y_train,y_pred))
print(accuracy_score(Y_test,test_preds))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_train)
print('Train accuracy score:',accuracy_score(Y_train,y_pred))
print('Test accuracy score:',accuracy_score(Y_test,knn.predict(X_test)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as RFC
rfc_b = RFC()
rfc_b.fit(X_train,Y_train)
y_pred = rfc_b.predict(X_train)
print('Train accuracy score:',accuracy_score(Y_train,y_pred))
print('Test accuracy score:', accuracy_score(Y_test,rfc_b.predict(X_test)))


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_train)
print('Train accuracy score:',accuracy_score(Y_train,y_pred))
print('Test accuracy score:', accuracy_score(Y_test,logreg.predict(X_test)))


# In[ ]:


lasso = Lasso()
lasso.fit(X_train,Y_train)
y_pred = lasso.predict(X_train)
print('Train accuracy score:',accuracy_score(Y_train,y_pred))
print('Test accuracy score:', accuracy_score(Y_test,lasso.predict(X_test)))

