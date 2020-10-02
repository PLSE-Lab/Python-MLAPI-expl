#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set()
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Dataset Retrieval 

#Training

Train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

#Testing
Test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(Train.shape)
print(Test.shape)


# ***Finding and Removing thge Missing Values***

# In[ ]:


TrainID = Train['Id']
TestID = Test['Id']

Train.drop("Id", axis = 1, inplace = True)
Test.drop("Id", axis = 1, inplace = True)

print(Train.shape)
print(Test.shape)


# In[ ]:


YTrain = Train.SalePrice.values


# In[ ]:


from scipy import stats
from scipy.stats import norm, skew

sns.distplot(Train['SalePrice'] , fit=norm);

plt.ylabel('Frequency')
plt.title('SalePrice distribution')

plt.show()


# In[ ]:


Train["SalePrice"] = np.log1p(Train["SalePrice"])

YTrain = Train.SalePrice.values

#Check the new distribution 
sns.distplot(Train['SalePrice'] , fit=norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()


# In[ ]:


TrainSize = Train.shape[0]
TestSize = Test.shape[0]


# In[ ]:


TrainAndTest = pd.concat((Train, Test), sort = False).reset_index(drop = True)
TrainAndTest.drop("SalePrice", axis = 1, inplace = True)


# In[ ]:


TrainAndTest.shape


# In[ ]:


#Finding thge Missing Values for XTrain
Missing_Percentage_Train_Test = TrainAndTest.isnull().sum().sort_values(ascending=False)/TrainAndTest.shape[0]
Missing_Columns_Values_Train_Test = Missing_Percentage_Train_Test[Missing_Percentage_Train_Test > 0]


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(x=Missing_Columns_Values_Train_Test.index.values, y=Missing_Columns_Values_Train_Test.values * 100, palette="Blues");
plt.title("Missing values Percentage XTrain");
plt.ylabel("%");
plt.xticks(rotation=75);


# In[ ]:


TrainAndTest[['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu']] = TrainAndTest[['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu']].fillna("None")


# In[ ]:


TrainAndTest[['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']] = TrainAndTest[['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']].fillna(0)


# In[ ]:


TrainAndTest[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','MSSubClass']]= TrainAndTest[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','MSSubClass']].fillna("None")


# In[ ]:


ModeFillerVariables = ['LotFrontage', 'MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']
for i in range(len(ModeFillerVariables)):
    TrainAndTest[ModeFillerVariables[i]]= TrainAndTest[ModeFillerVariables[i]].fillna(TrainAndTest[ModeFillerVariables[i]].mode()[0])


# In[ ]:


TrainAndTest["Functional"] = TrainAndTest["Functional"].fillna("Typ")


# In[ ]:


#Finding thge Missing Values for XTrain
Missing_Percentage_Train_Test = TrainAndTest.isnull().sum().sort_values(ascending=False)/TrainAndTest.shape[0]
Missing_Columns_Values_Train_Test = Missing_Percentage_Train_Test[Missing_Percentage_Train_Test > 0]


# In[ ]:


Missing_Columns_Values_Train_Test


# *** Finding the Categorical Columns ***

# In[ ]:


quantitative = [f for f in TrainAndTest.columns if TrainAndTest.dtypes[f] != 'object']
print("Quantitative: ",  len(quantitative))
qualitative = [f for f in TrainAndTest.columns if TrainAndTest.dtypes[f] == 'object']
print("Qualitative: ",  len(qualitative))
print(qualitative)


# In[ ]:


SkewedVariables = []
for i in range(len(quantitative)):
    if skew(TrainAndTest[quantitative[i]]) >= 1:
        SkewedVariables.append(quantitative[i])


# In[ ]:


TrainAndTest[SkewedVariables] = np.log1p(TrainAndTest[SkewedVariables])


# In[ ]:


TrainAndTest[qualitative] = OrdinalEncoder().fit_transform(TrainAndTest[qualitative])


# In[ ]:


qualitative = [f for f in TrainAndTest.columns if TrainAndTest.dtypes[f] == 'object']
print("Qualitative: ",  len(qualitative))


# *** Finding the corelation of features to target *** 

# In[ ]:


def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    newFeatures = []
    newFeatures = (spr.tail(15).feature).tolist() + (spr.head(5).feature).tolist()
    return newFeatures


# In[ ]:


newFeatures = spearman(Train, TrainAndTest.columns)


# In[ ]:


TrainAndTest = TrainAndTest[newFeatures]


# In[ ]:


TrainAndTest = pd.get_dummies(TrainAndTest)


# In[ ]:


Train = TrainAndTest[:TrainSize]
Test = TrainAndTest[TestSize+1:]


# In[ ]:


Train.shape


# In[ ]:


Test.shape


# # Splinting of Datasets -> Train and Test 

# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

import xgboost as xgb


# In[ ]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, Train, YTrain, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[ ]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(Train, YTrain)


# In[ ]:


rmse_cv(model_lasso).mean()


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(Train, YTrain)


# In[ ]:


xgb_preds = np.expm1(model_xgb.predict(Test))
lasso_preds = np.expm1(model_lasso.predict(Test))


# In[ ]:


preds = 0.7*lasso_preds + 0.3*xgb_preds


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = TestID
sub['SalePrice'] = preds
sub.to_csv('submission.csv',index=False)

