#!/usr/bin/env python
# coding: utf-8

# ## Simple analysis using XG Boost

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

import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# ### Reading Files

# In[ ]:


test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


pd.set_option('display.max_columns', 100)
train.head()


# In[ ]:


f = open('../input/house-prices-advanced-regression-techniques/data_description.txt','r')
message = f.read()
print(message)


# ### Checking NAs

# In[ ]:


NA_col = pd.DataFrame(train.isna().sum(), columns = ['NA_Count'])
NA_col['% of NA'] = (NA_col.NA_Count/len(train))*100
NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')

# Dropping 'PoolQC', 'MiscFeature', 'Alley', 'Fence'


# In[ ]:


pd.set_option('display.max_rows', 500)
NA_row = pd.DataFrame(train.isnull().sum(axis = 1), columns = ['NA_Row_Count'])
NA_row['% of NA'] = (NA_row.NA_Row_Count/len(train.columns))*100
NA_row.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')

# We are good row-wise


# In[ ]:


NA_col = pd.DataFrame(test.isna().sum(), columns = ['NA_Count'])
NA_col['% of NA'] = (NA_col.NA_Count/len(test))*100
NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')

# Dropping 'PoolQC', 'MiscFeature', 'Alley', 'Fence'


# In[ ]:


pd.set_option('display.max_rows', 500)
NA_row = pd.DataFrame(test.isnull().sum(axis = 1), columns = ['NA_Row_Count'])
NA_row['% of NA'] = (NA_row.NA_Row_Count/len(test.columns))*100
NA_row.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')

# We are good row-wise


# ### Dropping columns having >= 80% NAs

# In[ ]:


train_data = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1) 
test_data = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1) 


# ### Dropping the 'Id' column and the target variable to segregate Numerical and Categorical columns

# In[ ]:


train_wo_target = train_data.drop(['Id', 'SalePrice'], axis=1)


# In[ ]:


cols = train_wo_target.columns
num_cols = train_wo_target._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


# In[ ]:


num_cols


# In[ ]:


cat_cols


# In[ ]:


X_train = train_data.copy().drop('SalePrice', axis = 1)
y_train = train_data[['Id','SalePrice']]


# ### Imputation

# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer


# In[ ]:


# Imputing Numerical Columns

mean_imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
mean_imputer.fit(X_train[num_cols])
X_train[num_cols] = pd.DataFrame(mean_imputer.fit_transform(X_train[num_cols]))
test_data[num_cols] = pd.DataFrame(mean_imputer.fit_transform(test_data[num_cols]))


# In[ ]:


train_objs_num = len(X_train)
dataset = pd.concat(objs=[X_train, test_data], axis=0)
dataset_preprocessed = pd.get_dummies(dataset)
X_train = dataset_preprocessed[:train_objs_num]
test_data = dataset_preprocessed[train_objs_num:]


# In[ ]:


X_train.isna().sum()


# In[ ]:


test_data.isna().sum()


# ### Model Building - Gradient Boost

# In[ ]:


X_train.set_index('Id', inplace = True)
y_train.set_index('Id', inplace = True)
test_data.set_index('Id', inplace = True)


# In[ ]:


print(X_train.shape)
print(test_data.shape)


# In[ ]:


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', '## Reducing memory\nX_train = reduce_mem_usage(X_train)\ny_train = reduce_mem_usage(y_train)\ntest_data = reduce_mem_usage(test_data)')


# In[ ]:


from sklearn import metrics, tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBRegressor


# In[ ]:


XGB = XGBRegressor(n_jobs=-1)
 
# Use a grid over parameters of interest
param_grid = {
     'colsample_bytree': np.linspace(0.3, 0.8, 1.0),
     'n_estimators':[40 ,80, 150, 200,240, 300],
     'max_depth': [1,2, 3,5, 8]
}

 
CV_XGB = GridSearchCV(estimator=XGB, param_grid=param_grid, cv= 10)


# In[ ]:


# Train XGBoost Regressor
get_ipython().run_line_magic('time', 'CV_XGB.fit(X_train, y_train)')


# In[ ]:


best_xgb_model = CV_XGB.best_estimator_
print (CV_XGB.best_score_, CV_XGB.best_params_)


# In[ ]:


pred_train_xgb = best_xgb_model.predict(X_train)
pred_test_xgb = best_xgb_model.predict(test_data)


# In[ ]:


print(metrics.mean_squared_log_error(y_train, pred_train_xgb).round(5))


# In[ ]:


y_pred_test_xgb = pd.DataFrame(pred_test_xgb, columns = ['SalePrice'])
y_pred_test_xgb['Id'] = test['Id']


# In[ ]:


columnsTitles = ['Id', 'SalePrice']

submission = y_pred_test_xgb.reindex(columns=columnsTitles)
submission .head()


# In[ ]:


filename = 'House_Pricing_XGB.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

