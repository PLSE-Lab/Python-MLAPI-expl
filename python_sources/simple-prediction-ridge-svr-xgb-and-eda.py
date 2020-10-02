#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Basic
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Stat
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from scipy.stats import norm, skew #for some statistics

#Modeling

from sklearn.svm import SVR
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

#Test
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report,confusion_matrix

import os


# In[ ]:


pwd


# In[ ]:


df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


corr=df.corr()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0,cmap="RdBu_r",square=True)


# In[ ]:


df.info()


# # DATA PROCESSING

# # Outlilers

# In[ ]:


sns.scatterplot(x=df['OverallQual'],y=df['SalePrice'])


# In[ ]:


sns.scatterplot(x=df['GrLivArea'],y=df['SalePrice'])


# In[ ]:


df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)


# # Reponse Variable

# In[ ]:


sns.distplot(df['SalePrice'])


# In[ ]:


stats.probplot(df['SalePrice'],plot=plt)
plt.show()


# In[ ]:


df['SalePrice'],fitted_lambda = stats.boxcox(df['SalePrice'])


# In[ ]:


fitted_lambda


# In[ ]:


sns.distplot(df['SalePrice'])


# # Feature Engineering

# In[ ]:


y = df['SalePrice'].reset_index(drop=True)
df_train = df.drop(['SalePrice'], axis=1)
df_test = test


# In[ ]:


df_all = pd.concat([df_train, df_test]).reset_index(drop=True)
df_all.shape


# In[ ]:


missing_data = df_all.isnull().sum().sort_values(ascending=False)
n=[ ]
for i in range(len(missing_data)):
    if missing_data[i]==0:
        n.append(i)


# In[ ]:


missing_data[0:min(n)]


# In[ ]:


#'PoolQC'


# In[ ]:


df_all['PoolQC'] = df_all['PoolQC'].fillna("None")


# In[ ]:


#'MiscFeatures'


# In[ ]:


df_all["MiscFeature"] = df_all["MiscFeature"].fillna("None")


# In[ ]:


#'Alley'


# In[ ]:


df_all["Alley"] = df_all["Alley"].fillna("None")


# In[ ]:


#'Fence'


# In[ ]:


df_all["Fence"] = df_all["Fence"].fillna("None")


# In[ ]:


#'FireplaceQu'


# In[ ]:


df_all["FireplaceQu"] = df_all["FireplaceQu"].fillna("None")


# In[ ]:


#'LotFrontage'


# In[ ]:


df_all['LotFrontage']=df_all.groupby(by="Neighborhood")['LotFrontage'].transform (lambda x: x.fillna(x.median()))


# In[ ]:


#'GarageCond, GarageType, GarageYrBlt, GarageFinish, GarageQual'


# In[ ]:


for x in ('GarageCond', 'GarageType', 'GarageFinish', 'GarageQual'):
    df_all[x] = df_all[x].fillna('None')


# In[ ]:


#'GarageYrBlt'


# In[ ]:


df_all['GarageYrBlt'] = df_all['GarageYrBlt'].fillna(0)


# In[ ]:


#'BsmtFinType2, BsmtExposure, BsmtQual, BsmtFinType1, BsmtCond'


# In[ ]:


for x in ('BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtCond'):
    df_all[x] = df_all[x].fillna('None')


# In[ ]:


#'MasVnrType','MasVnrArea'


# In[ ]:


df_all["MasVnrType"] = df_all["MasVnrType"].fillna("None")
df_all["MasVnrArea"] = df_all["MasVnrArea"].fillna(0)


# In[ ]:


#"Electrical"


# In[ ]:


df_all["Electrical"]=df_all['Electrical'].fillna("SBrkr")


# In[ ]:


#MSZoning


# In[ ]:


df_all['MSZoning'] = df_all['MSZoning'].fillna(df_all['MSZoning'].mode()[0])


# In[ ]:


#Functional


# In[ ]:


df_all["Functional"] = df_all["Functional"].fillna("Typ")


# In[ ]:


#Utilities


# In[ ]:


df_all = df_all.drop('Utilities', axis=1)


# In[ ]:


#'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'


# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_all[col] = df_all[col].fillna(0)


# In[ ]:


#SaleType


# In[ ]:


df_all['SaleType']=df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])


# In[ ]:


#'GarageArea', 'GarageCars'


# In[ ]:


for col in ('GarageArea', 'GarageCars'):
    df_all[col] = df_all[col].fillna(0)


# In[ ]:


#'Exterior1st','Exterior2nd'


# In[ ]:


df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])
df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])


# In[ ]:


#'KitchenQual'


# In[ ]:


df_all['KitchenQual'] = df_all['KitchenQual'].fillna(df_all['KitchenQual'].mode()[0])


# # Check features of skewness

# In[ ]:


numeric_features = df_all.dtypes[df_all.dtypes != "object"].index


# In[ ]:


numeric_features


# In[ ]:


skewness = pd.DataFrame({'variable':numeric_features,'skew':skew(df_all[numeric_features])})
skewness.sort_values('skew',ascending=False).head(10) 


# In[ ]:


feature_skew = skewness[skewness['skew']>1].index
numeric_features[feature_skew]


# In[ ]:


for x in numeric_features[feature_skew]:
    df_all[x]=np.log1p(df_all[x])


# # Get Dummies

# In[ ]:


dummies = df_all.dtypes[df_all.dtypes=='object'].index
dummies


# In[ ]:


df_all = pd.get_dummies(df_all,drop_first=True)


# # Create Train and Test data set

# In[ ]:


X = df_all.iloc[:len(y),:]
X_test = df_all.iloc[len(y):,:]


# In[ ]:


X.shape


# In[ ]:


X_test.shape


# In[ ]:


y.shape


# # Modeling

# In[ ]:


kf = KFold(n_splits=12, random_state=42, shuffle=True)


# In[ ]:


def rmsle(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# In[ ]:


#Ridge regression


# In[ ]:


ridge_alphas = [0.1, 1.0, 10.0, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))


# In[ ]:


#SVM


# In[ ]:


svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))


# In[ ]:


#XGB


# In[ ]:


XGB = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


# # Model Train

# In[ ]:


# Ridge regression


# In[ ]:


scores = {}
score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())


# In[ ]:


ridge_model = ridge.fit(X,y)


# In[ ]:


# SVM regression


# In[ ]:


score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())


# In[ ]:


svm_model = svr.fit(X,y)


# In[ ]:


# XGB


# In[ ]:


score = cv_rmse(XGB)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())


# In[ ]:


XGB_model = XGB.fit(X,y)


# In[ ]:


# Blended Model


# In[ ]:


def Blended_Machine(X):
    return ((0.4 * ridge_model.predict(X)) +(0.3 * svm_model.predict(X)) + (0.3 * XGB_model.predict(X))) 


# In[ ]:


blended_score = rmsle(y, Blended_Machine(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)


# # Submit

# In[ ]:


log_prediction = Blended_Machine(X_test)


# In[ ]:


print(log_prediction.shape)


# In[ ]:


prediction = inv_boxcox(log_prediction, -0.07712954824421477)


# In[ ]:


submission = pd.DataFrame()


# In[ ]:


submission['Id']=X_test['Id']


# In[ ]:


submission['SalePrice'] = prediction


# In[ ]:


prediction


# In[ ]:


submission.to_csv('submit1.csv',index=False)


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

