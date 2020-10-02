#!/usr/bin/env python
# coding: utf-8

# # Regression

# Basic Libraries

# In[ ]:


#Importing the libraries
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA, TruncatedSVD

from math import sqrt
import warnings
warnings.filterwarnings('ignore')


# Loading Data

# In[ ]:


#Reading data files
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
sample_path = '../input/sample_submission.csv'
sample = pd.read_csv(sample_path)
RANDOM_STATE=0  #to make use of in train-test split as well as model for randomness


# Feature Engineering

# In[ ]:


#Dropping features with high missing values
train_clean=home_data.drop(columns=['MiscFeature','Fence','PoolQC','FireplaceQu','Alley'])


# In[ ]:


#Creating features and labels
X=train_clean.drop(columns=['SalePrice'])
y=home_data[['SalePrice']]


# In[ ]:


#Split the dataset to evaluate out sample performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.15, random_state=RANDOM_STATE)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


#Seperation of numerical and categorical features
num_feat=X_train.select_dtypes(include='number').columns.to_list()
cat_feat=X_train.select_dtypes(exclude='number').columns.to_list()


# Pipeline

# In[ ]:


num_pipe=Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe=Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
ct=ColumnTransformer(remainder='drop',
                    transformers=[
                        ('numerical', num_pipe, num_feat),
                        ('categorical', cat_pipe, cat_feat)
                    ])
model=Pipeline([
    ('transformer', ct),   
    ('predictor', GradientBoostingRegressor())
])


# In[ ]:


model.fit(X_train, y_train);


# Prediction

# In[ ]:


y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)


# Evaluation of Result

# In[ ]:


print('In sample MAE error: ', round(mean_absolute_error(y_pred_train, y_train)))
print('Out sample MAE error: ', round(mean_absolute_error(y_pred_test, y_test)))


# In[ ]:


#model.fit(X,y);


# Result Submission

# In[ ]:


def submission(test, model):
    y_pred=model.predict(test)
    result=pd.DataFrame({'Id':sample.Id, 'SalePrice':y_pred})
    result.to_csv('/kaggle/working/result.csv',index=False)


# In[ ]:


submission(test_data, model)


# In[ ]:


check=pd.read_csv('/kaggle/working/result.csv')
check.head()

