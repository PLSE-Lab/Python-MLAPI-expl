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


#Import libraries
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew, norm
from scipy.stats.stats import pearsonr
from scipy import stats
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
train.head()


# In[ ]:


test.head()


# **Data Cleaning**

# In[ ]:


#Check training data for missing values
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))


# In[ ]:


#DEAL WITH MISSING VALUES

#Delete columns from both training and test data where missing data is more than 1 (all except Electrical)
train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)

#Delete observation with missing Electrical data from training data
train = train.drop(train.loc[train['Electrical'].isnull()].index)

#Need to redefine SalePrice to account for missing observation
y = train['SalePrice'];  

#Check all missing values are gone
print(train.isnull().sum().max())


# In[ ]:


train.shape


# In[ ]:


#Drop same columns from test data
test = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',
       'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',
       'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',
       'MasVnrArea', 'MasVnrType'], 1)


# In[ ]:


test.shape


# In[ ]:


#Look for outlier - I know from previous kernel GrLivArea has a couple
var = 'GrLivArea'
data = pd.concat((train[var], y), axis=1)
data.plot.scatter(x=var, y='SalePrice');


# In[ ]:


#Identify IDs of outliers based on above charts - 2 in GrLivArea, 1 in TotalBsmtSF
print(train.sort_values(by = 'GrLivArea', ascending = False)[:2])
print(train.sort_values(by = 'TotalBsmtSF', ascending = False)[:1])


# In[ ]:


#Delete them
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)


# In[ ]:


#Get target variable
y = train['SalePrice'].copy()
y.head()


# In[ ]:


#Combine Kaggle's train and test data for transformations
all_data = pd.concat([train, test], sort=False)
all_data = all_data.drop(['SalePrice'], 1)
print(all_data.shape)
all_data.head()


# In[ ]:


#Check target variable for skew
sns.distplot(y, fit=norm);
fig = plt.figure()
res = stats.probplot(y, plot=plt)


# In[ ]:


#Resolve skew in target variable by taking logs
y = np.log1p(y)
sns.distplot(y, fit=norm);
fig = plt.figure()
res = stats.probplot(y, plot=plt)


# In[ ]:


#Isolate numeric features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#Compute skewness
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) 
#Set threshold for skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
#Identify skewed numeric features
skewed_feats = skewed_feats.index
print(skewed_feats)

#Log transform combined dataset features
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data.head()


# In[ ]:


#Get dummies for categorial variables - note get_dummies only works on strings
all_data = pd.get_dummies(all_data)
all_data.head()


# In[ ]:


all_data = all_data.fillna(0)
all_data = all_data.drop(['Id'], 1)
all_data.head()


# In[ ]:


#Refresh Kaggle train and test datasets with log transformed numeric features
Xtrain = all_data[:train.shape[0]]
Xtest = all_data[train.shape[0]:]
print(Xtrain.shape)
print(Xtest.shape)
Xtrain.head()


# In[ ]:


#Feature Scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain.values)
Xtrain_scaled_df = pd.DataFrame(Xtrain_scaled, index = Xtrain.index, columns = Xtrain.columns)
Xtest_scaled = scaler.transform(Xtest.values)
Xtest_scaled_df = pd.DataFrame(Xtest_scaled, index = Xtest.index, columns = Xtest.columns)
Xtrain = Xtrain_scaled_df
Xtest = Xtest_scaled_df


# In[ ]:


Xtrain.head()


# **Modeling**

# In[ ]:


#DEFINE ERROR FUNCTION: RMSE
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, Xtrain, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[ ]:


model_ridge = Ridge()


# In[ ]:


#Ridge Regression

#Tune alpha (regularization parameter)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[ ]:


cv_ridge = pd.Series(cv_ridge, index=alphas)
cv_ridge.plot(title="Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")


# **Prepare predictions to submit**

# In[ ]:


cv_ridge.min()


# In[ ]:


#Lasso
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(Xtrain,y)


# In[ ]:


rmse_cv(model_lasso).mean()


# In[ ]:


coef = pd.Series(model_lasso.coef_, index=Xtrain.columns)


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")


# In[ ]:


#Make predictions
lnTestY = model_lasso.predict(Xtest)
TestY = np.expm1(lnTestY)
TestY = pd.Series(TestY)
print(TestY.shape)
print(TestY.head())


# In[ ]:


#Prepare submission file
submission = pd.DataFrame({'Id':test['Id'], 'SalePrice':TestY})
print(submission.head())
print(submission.shape)


# In[ ]:


submission.to_csv('hpsubmission.csv', index=False)

