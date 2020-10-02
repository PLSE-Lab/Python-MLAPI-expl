#!/usr/bin/env python
# coding: utf-8

# # Kaggle - House Prices / Advanced Regression Techniques

# This notebook deals with the Kaggle competition "House Prices: Advanced Regression Techniques" (https://www.kaggle.com/c/house-prices-advanced-regression-techniques) and generates a response dataset that, once submitted, gets a score below 0.12 RMSLE. It is possible to obtain better results using ensemble methods encapsulating several algorithms. However the purpose of this notebook is to be able to tune efficiently the hyperparameters of a linear regression algorithm in order to get a good result.

# ## 1. Preprocessing

# ### Loading modules and data

# We start by loading the necessary libraries

# In[ ]:


#basic modules
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge


# We now load the data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')

X = df_train.drop(['SalePrice', 'Id'], axis = 1)
train_labels = df_train['SalePrice'].values
X_test = df_test.drop(['Id'], axis = 1)


# We show the type of each column

# In[ ]:


#df_data_types = X.dtypes

for colname in X.columns:
    print(colname + ': ' + str(X[colname].dtype))


# as we see, the variable MSSubClass has the wrong type, since it should be categorical. We change its type to string in both the training and test set

# In[ ]:


def prepro_change_column_type(x):
    x['MSSubClass'] = x['MSSubClass'].astype('str')

    
prepro_change_column_type(X)

prepro_test_list =[]
prepro_test_list.append(prepro_change_column_type)
    
# df_train['MSSubClass'] = df_train['MSSubClass'].astype('str')
# df_test['MSSubClass']  = df_test['MSSubClass'].astype('str')


# ### Objective treatment of missing values

# We remove variables with more than 50% of missing values since they will not add value to the dataset

# In[ ]:


#proportion of NaN in each column
count_var_nulls = X.isnull().sum(axis = 0)/X.shape[0]
variable_nulls = count_var_nulls[count_var_nulls >0]

print('variables with NaN:')
print(variable_nulls)
print('-----')
#remove columns with more than 50% of NaN
remove_variables_index = list(variable_nulls[variable_nulls > 0.5].index)
variable_nulls.drop(remove_variables_index, inplace = True)  #prepro remove_variables_index

def prepro_nan_columns(x):
    x.drop(remove_variables_index, axis =1,  inplace = True)

print('remaining variables with NaN after dropping those with more than 50% missing:')
print(variable_nulls)  

prepro_nan_columns(X)
prepro_test_list.append(prepro_nan_columns)


# We create lists with the remaining column names for numeric and categorical variables, for later reference

# In[ ]:


num_columns = X.select_dtypes(include=np.number).columns.tolist() 
cat_columns = X.select_dtypes(include=['object', 'category']).columns.tolist() 


# We remove observations with more than 50% of missing values since they will not add value to the dataset

# In[ ]:


count_obs_nulls = X.isnull().sum(axis = 1)/X.shape[1]
obs_nulls = count_obs_nulls[count_obs_nulls >0]
remove_obs_index = list(obs_nulls[obs_nulls > 0.5].index)
X.drop(remove_obs_index, axis = 1, inplace = True)
print(len(remove_obs_index),' observations removed because of having more than 50% of null values' )


# ### Objective Imputation of NaN in variables

# There are a few cases where missing values can be imputed objectively from the knowledge of this particular dataset. We will do that for these cases instead of using an automatic imputation strategy at a later stage.

# In[ ]:


#prepro
def prepro_nan_objective_imputing(x):
    #categorical
    x['MasVnrType'].fillna('None', inplace = True)
   
    aux_list = ['BsmtQual', 
                  'BsmtCond', 
                  'BsmtExposure',
                  'BsmtFinType1',
                  'BsmtFinType2',
                  'GarageType', 
                  'GarageFinish',
                  'GarageQual', 
                  'GarageCond',
                  'FireplaceQu']
    
    for i in aux_list:
        x[i].fillna('NA', inplace = True)  
        
    #numerical
    x['MasVnrArea'].fillna(0, inplace = True)    
    
  
       
    x.loc[x['BsmtCond'] == 'NA', ['BsmtUnfSF', 
                                   'TotalBsmtSF', 
                                   'BsmtFullBath', 
                                   'BsmtHalfBath', 
                                   'BsmtFinSF1', 
                                   'BsmtFinSF2' ]] = 0
    
prepro_nan_objective_imputing(X)
prepro_test_list.append(prepro_nan_objective_imputing)


# ### Outliers for numerical variables

# We now look at outliers. We will remove a few of the rows presenting outliers in the train set for fitting a more robust model. The following code chunk (need to uncomment) generates a histogram for each numerical variable that is of assistance in choosing appropriate thresholds for declaring values as outliers.

# In[ ]:


X = X.loc[
    (  X['LotArea']<100000) 
    | (X['LotFrontage']<250)
    | (X['1stFlrSF']<4000)
    | (X['BsmtFinSF1']<5000) 
    | (X['BsmtFinSF2']<1400) 
    | (X['EnclosedPorch']<500)
    | (X['GrLivArea']<5000)
    | (X['TotalBsmtSF']<6000), ]


# ### Automatic missing values imputation and normalization

# We now prepare transformers for automatic imputation of missing values in numeric and categorical variables. The strategies will be the median and most frequent value imputation respectively. After that We will standardize the numeric variables and one-hot encode the categorical ones.
# 
# Since we are going to fit our model using cross validation we cannot do this in an objective way with the original dataset, since for each cross validation round these median and most common value may vary. Hence the need for an automatic process that can be included in a pipeline.
# 

# In[ ]:


numeric_transformer = Pipeline([
                                ("median", SimpleImputer(strategy='median')),
                                ("standard", StandardScaler())
                               ]
                              )
categorical_transformer = Pipeline([
                                ("mostfreq", SimpleImputer(strategy = 'most_frequent')),
                                ("onehot", OneHotEncoder(handle_unknown='ignore'))
                                   ]
                                  )


# In[ ]:


prepro_transformer = ColumnTransformer([('num', numeric_transformer, num_columns),
                                       ('cat', categorical_transformer, cat_columns)])


# ## 2. Modeling

# We log-transform the target variable since the metric used by Kaggle for this competition is RMSLE and create a pipeline encapsulating the preprocessing steps and the final estimator that will be used.

# In[ ]:


y = np.log10(train_labels)


# In[ ]:


pipeline = Pipeline([('prepro', prepro_transformer), ('estimator', Ridge())])


# ### Choosing models and tuning parameters

# We apply a Grid Search approach to find optimal hyperparameters for the algorithm.

# In[ ]:


param_grid = dict(estimator__alpha =  np.linspace(10,100, 100))
grid_search = GridSearchCV(pipeline, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 0)


# In[ ]:


grid_search.fit(X, y)


# In[ ]:


best_model = grid_search.best_estimator_


# ## 3 Predicting

# We first preprocess the test set in the same way as the train set

# In[ ]:


for func in prepro_test_list:
    func(X_test)


# We make the predictions and prepare the csv file for submitting to Kaggle

# In[ ]:


y_pred   = 10**best_model.predict(X_test)
df_submit = pd.DataFrame({'Id':df_test['Id'], 'SalePrice': y_pred})
df_submit.to_csv('df_submit.csv', index = False)


# In[ ]:


grid_search.best_params_

