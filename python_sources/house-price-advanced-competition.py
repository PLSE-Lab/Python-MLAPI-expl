#!/usr/bin/env python
# coding: utf-8

# # My Imports

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.impute import SimpleImputer


from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
print('Compelete Imports')


# # Import my dataSet Train_test

# In[ ]:


#File Path Full
file_path_train='../input/house-prices-advanced-regression-techniques/train.csv'
file_path_test='../input/house-prices-advanced-regression-techniques/test.csv'

# Read the train data
dp_train = pd.read_csv(file_path_train, index_col='Id')
# Read the test data
dp_test=pd.read_csv(file_path_test,index_col='Id')
# Remove rows with missing target, separate target from predictors

dp_train.dropna(axis=0, subset=['SalePrice'], inplace=True)
#define target
y = dp_train.SalePrice
#Remove target colum
X = dp_train.drop(['SalePrice'], axis=1)
X_test=dp_test

# To keep things simple, we'll use only numerical predictors
# X = dp_train.select_dtypes(exclude=['object'])
# X_test = X_test_full.select_dtypes(exclude=['object'])
# # To keep things simple, we'll use only numerical predictors
# X = X_full.select_dtypes(exclude=['object'])
# X_test = dp_test.select_dtypes(exclude=['object'])


print('Read Done!')
# X_test.head()

# X.head()


# ## Split 

# In[ ]:


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

print('Split Done!')


# ## count columns with missing values 

# In[ ]:


# X_train.isna().sum()


# In[ ]:



pd.concat([dp_train.isnull().sum(),
          100* dp_train.isnull().sum()/len(dp_train)],
         axis=1).rename(columns={0:'Missing Records',1:'Percentage (%)'})


# In[ ]:


category_colums=[colum for colum in X.columns if X[colum].dtype == 'object']
len(category_colums)


# In[ ]:


y.isna().sum()


# In[ ]:


X_train.isna().sum()


# #### get columns with missing values

# In[ ]:


cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
len(cols_with_missing)


# In[ ]:


# # Number of missing values in each column of training data
# missing_val_count_by_column = (X_train.isnull().sum())
# print(missing_val_count_by_column[missing_val_count_by_column > 0])
# missing_val_count_by_column[missing_val_count_by_column > 0].shape


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# # Imputation for Messing Values

# In[ ]:


bad_cols = [ col for col in category_colums if set(X[col].unique()) != set(X_test[col].unique()) ]
bad_cols


# In[ ]:


len(bad_cols)


# In[ ]:


X.drop(columns=bad_cols,axis=1,inplace=True)
X_test.drop(columns=bad_cols,axis=1,inplace=True)


# In[ ]:


category_colums=[colum for colum in X.columns if X[colum].dtype == 'object']
len(category_colums)


# In[ ]:


from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer(strategy='most_frequent')
imputed_X=pd.DataFrame(my_imputer.fit_transform(X))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

# Imputation removed column names; put them back
imputed_X.columns = X.columns
imputed_X_test.columns = X_test.columns


# In[ ]:


imputed_X.head()


# In[ ]:


imputed_X_test.head()


# # Label Encoding for Hundel Categories

# In[ ]:


# Make copy to avoid changing original data 
imputed_X_lable = X.copy()
imputed_X_test_lable = X_test.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in category_colums:
    imputed_X_lable[col] = label_encoder.fit_transform(imputed_X[col])
    imputed_X_test_lable[col] = label_encoder.transform(imputed_X_test[col])


# In[ ]:


imputed_X_lable


# In[ ]:


imputed_X_test_lable


# In[ ]:


imputed_X_lable.dtypes


# # Setup hyper-parameters

# In[ ]:


parameters = [{
    'n_estimators': list(range(100, 1001, 100)), 
    'learning_rate': [x / 100 for x in(range(5, 100, 10))], 
    'max_depth': list(range(6, 70, 6))
}]
parameters


# # Use Grid_Search_CV to get Best Parameters
# 

# In[ ]:



gsearch = GridSearchCV(estimator=XGBRegressor(),
                       param_grid = parameters, 
                       scoring='neg_mean_absolute_error',
                       n_jobs=4,
                       cv=5,
                       verbose=7)

gsearch.fit(imputed_X_lable,y)

gsearch.best_params_.get('n_estimators'),gsearch.best_params_.get('learning_rate'),gsearch.best_params_.get('max_depth')


# ## Get Best Parameters 

# In[ ]:


best_n_estimators=gsearch.best_params_.get('n_estimators')
best_learning_rate=gsearch.best_params_.get('learning_rate')
best_max_depth=gsearch.best_params_.get('max_depth')

best_n_estimators,best_learning_rate,best_max_depth


# # Best Model

# In[ ]:


best_model=XGBRegressor(n_estimators=best_n_estimators,
                       learning_rate=best_learning_rate,
                       max_depth=best_max_depth)
best_model.fit(imputed_X_lable,y)


# ## Prediction 

# In[ ]:


my_preds_test=best_model.predict(imputed_X_test_lable)
my_preds_test


# # Make output

# In[ ]:


my_output=pd.DataFrame(
{'Id':X_test.index,
'SalePrice':my_preds_test})
my_output.to_csv('submission.csv',index=False)
print('create output done!')

