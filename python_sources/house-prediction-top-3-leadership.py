#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.isnull().sum().sort_values(ascending=False)[:19]


# In[ ]:


test.shape


# In[ ]:


test


# In[ ]:


test.isnull().sum().sort_values(ascending=False)[:33]


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False, cmap='plasma')


# In[ ]:


test.LotFrontage.isnull().sum()


# In[ ]:


train.LotFrontage.isnull().sum()


# In[ ]:


train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())


# In[ ]:


train.LotFrontage.isnull().sum()


# In[ ]:


test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())


# In[ ]:


test['LotFrontage'].isnull().sum()


# In[ ]:


train.Alley.value_counts(dropna=False)


# In[ ]:


train.drop(columns=['Alley'], inplace = True)


# In[ ]:


train


# In[ ]:


test.Alley.value_counts(dropna = False)


# In[ ]:


test.drop(columns=['Alley'], inplace = True)


# In[ ]:


test


# In[ ]:



list1=['BsmtQual', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 'MasVnrArea',
         'BsmtExposure','BsmtFinType2','BsmtCond']

for item in list1:
    train[item] = train[item].fillna(train[item].mode()[0])
    test[item] = test[item].fillna(test[item].mode()[0])


# In[ ]:


train.isnull()


# In[ ]:


list1 = ['GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature']

for item in list1:
    train.drop(columns=item, inplace=True)
    test.drop(columns=item, inplace=True)


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train.drop(columns=['Id'], inplace = True)


# In[ ]:


train


# In[ ]:


train.shape


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])


# In[ ]:


columns = ['BsmtFinType1', 'Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 
           'Exterior1st', 'KitchenQual']
columns1 = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea']

for item in columns:
    test[item] = test[item].fillna(test[item].mode()[0])
for item in columns1:
    test[item] = test[item].fillna(test[item].mean())


# In[ ]:


test.drop(columns=['Id'], inplace=True)


# In[ ]:


test.shape


# In[ ]:


test.isnull().any().any()


# In[ ]:


train.isnull().any().any()


# In[ ]:


train.LotConfig.value_counts(dropna = False)


# In[ ]:


train.LotConfig.unique()


# In[ ]:


columns = ['MSZoning', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
       'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']


# In[ ]:


len(columns)


# In[ ]:


final_df=pd.concat([train,test], axis=0)


# In[ ]:


final_df.shape


# In[ ]:


def One_hot_encoding(columns):
    df_final=final_df
    i=0
    for fields in columns:
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:           
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[ ]:


final_df = One_hot_encoding(columns)


# In[ ]:


final_df.shape


# In[ ]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[ ]:


final_df.shape


# In[ ]:


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[ ]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[ ]:


X_train


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [100, 500, 900]
criterion = ['gini', 'entropy']
depth = [3,5,10,15]
min_split=[2,3,4]
min_leaf=[2,3,4]
bootstrap = ['True', 'False']
verbose = [5]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':depth,
    'criterion':criterion,
    'bootstrap':bootstrap,
    'verbose':verbose,
    'min_samples_split':min_split,
    'min_samples_leaf':min_leaf
    }

random_cv = RandomizedSearchCV(estimator=regressor,
                               param_distributions=hyperparameter_grid,
                               cv=5, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = 4, 
                               return_train_score = True,
                               random_state=42)


# In[ ]:


random_cv.fit(X_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor = RandomForestClassifier(bootstrap='False', class_weight=None,
                       criterion='entropy', max_depth=10, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=3,
                       min_samples_split=3, min_weight_fraction_leaf=0.0,
                       n_estimators=900, n_jobs=None, oob_score=False,
                       random_state=None, verbose=5, warm_start=False)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(df_Test)


# In[ ]:


y_pred


# In[ ]:


pred=pd.DataFrame(y_pred)
samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


sub = pd.concat([samp['Id'],pred], axis=1)
sub.columns=['Id','SalePrice']


# In[ ]:


sub


# In[ ]:


import xgboost


# In[ ]:


regressor=xgboost.XGBRegressor()


# In[ ]:


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[ ]:


random_cv.fit(X_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


regressor = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1, missing=None, monotone_constraints='()',
             n_estimators=900, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(df_Test)


# In[ ]:


y_pred


# In[ ]:


pred=pd.DataFrame(y_pred)
samp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.concat([samp['Id'],pred], axis=1)
sub.columns=['Id','SalePrice']


# In[ ]:


sub


# In[ ]:


#sub.to_csv('My_sub1.csv',index=False)


# In[ ]:




