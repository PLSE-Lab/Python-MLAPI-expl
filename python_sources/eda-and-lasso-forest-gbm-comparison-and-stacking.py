#!/usr/bin/env python
# coding: utf-8

# # Explanatory data analysis

# In[230]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# In[309]:


df_train = pd.read_csv('../input/train.csv').drop('Id',axis=1)
#removing outliers recomended by the author of the dataset
df_train = df_train[df_train["GrLivArea"]<4500]


# In[282]:


df_train.head()


# There is a mixture of numerical and categorical features.

# In[283]:


sum(df_train.isna().sum()!=0)


# There are 19 categories with missing values.

# In[284]:


missing = df_train.isna().sum()
missing = missing[missing > 0].sort_values()
missing.plot.barh();


# The [data description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) indicates most of the missing features correspond to no feature found. For imputing categorical variables we will create a 'None' category.

# In[285]:


print('Categorical variables:',sum(df_train.dtypes=='O'))
print('Categorical variables:',sum(df_train.dtypes!='O'))


# There are 43 variables encoded as categorical ('object' type) and 37 variables encoded as numerical type. Reading the data description indicates that **MSSubClass** is encoded as numerical but is actually categorical.

# In[286]:


plt.hist(df_train['SalePrice']);


# The **SalePrice** variable is not normally distributed.

# In[313]:


plt.hist(np.log(df_train['SalePrice']));


# Log transform is acceptable for converting **SalePrice** to a normally distributed variable.

# In[287]:


sns.heatmap(df_train.fillna(0).corr().sort_values('SalePrice').sort_values('SalePrice',axis=1).iloc[-11:,-11:]);


# The highly correlated features with **SalePrice** have correlation with other features. Several features in the top-10 list are related to area of the property or area of the house. We will use this for feature engineering.

# # Preprocessing/feature engineering

# In[288]:


from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
def fixskew(data):
    return boxcox1p(data,boxcox_normmax(data+1))


# In[311]:


# create normally distributed independent variable
y = np.log(df_train['SalePrice'])

# combine test and train samples for features engineering
df_test = pd.read_csv('../input/test.csv').drop('Id',axis=1)
features = df_train.drop('SalePrice',axis=1).append(df_test)

# drop features chosen by hand
features = features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)

# create new year feature
features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

# create new total square footage features
features['TotalSF1']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

# create new square footage feature ingnore unfinished basement
features['TotalSF2'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

# create total number of bathroom feature
features['TotalBath'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

# create porch square footage feature
features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])

# create onehot features for existence of several features
features['HasPool'] = (features['PoolArea']!=0).astype('int')
features['Has2ndFloor'] = (features['2ndFlrSF']!=0).astype('int')
features['HasGarage'] = (features['GarageArea']!=0).astype('int')
features['HasBsmt'] = (features['TotalBsmtSF']!=0).astype('int')
features['HasFireplace'] = (features['Fireplaces']!=0).astype('int')

# make numerical features normal
numcols = features.columns[features.dtypes!='O']
for col in numcols:
    features[col]=features[col].fillna(features[col].mean())

# encode categorical features using target mapping
catcols = features.columns[features.dtypes=='O']
for col in catcols:
    target_map = df_train['SalePrice'].groupby(df_train[col].fillna('None')).mean()
    features[col+'_target']=features[col].map(target_map).fillna(df_train['SalePrice'].mean())

# also encode categorical features using onehot
features = pd.get_dummies(features.fillna('None'))

# split features into train and test
X = features[:len(y)]
X_test = features[len(y):]

# remove outliers
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

# remove onehot categories with more than 99% zeros to prevent overfitting
overfit = []
for col in X.columns:
    counts = X[col].value_counts()
    num_zeros = counts.iloc[0]
    if num_zeros / len(X) * 100.0 > 99.0:
        overfit.append(col)
X = X.drop(overfit, axis=1)
X_test = X_test.drop(overfit, axis=1)


# # Modeling

# In[274]:


from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from mlxtend.regressor import StackingRegressor

kf = KFold(n_splits=5,shuffle=True)
def scoreit(model,X=X):
    return(np.sqrt(-cross_val_score(model,X,y,cv=kf,scoring='neg_mean_squared_error')))


# In[312]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha=3e-4))
scores = scoreit(lasso,X)
print("RMSE:",scores.mean(),"+/-",scores.std())


# We will use lasso as a baseline for comparison with other methods. Our cross validation score is good compared with the leaderboard values. The value of alpha was chosen manually. Performance on the test data will be worse. Also, we are measuring root mean square error of logs, not root mean square log error.

# In[279]:


randomforest = RandomForestRegressor(n_estimators=100)
scores = scoreit(randomforest,X)
print("RMSE:",scores.mean(),"+/-",scores.std())


# Random forest does not perform well on this dataset. The score is lower and the training time is longer. Increasing n_estimators will increase performance but training time scales linearly with n_estimators.

# In[ ]:


xgb = make_pipeline(RobustScaler(), XGBRegressor(
    learning_rate=0.01, n_estimators=3400, max_depth=3, min_child_weight=0,
    gamma=0, subsample=0.75, colsample_bytree=0.7, objective='reg:linear', nthread=-1,
    scale_pos_weight=1, reg_alpha=0.00005))
scores = scoreit(xgb,X)
print("RMSE:",scores.mean(),"+/-",scores.std())


# Gradient boosting performs well but not significantly better than lasso.

# In[ ]:


stack = StackingRegressor([lasso,xgb,randomforest],meta_regressor=xgb)
scores = scoreit(stack,X)
print("RMSE:",scores.mean(),"+/-",scores.std())


# Stacking does not improve performance. Lasso, gradient boosting, and stacking all perform similarly with cross val score.

# # Final prediction

# We will use lasso for the final submission.

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha=3e-4))
lasso.fit(X,y)
pred = pd.Series(np.exp(lasso.predict(X_test)))

test_id = pd.read_csv('../input/test.csv')['Id']
output = pd.DataFrame({'Id':test_id, 'SalePrice':pred})
output.to_csv('submission.csv', index=False)

