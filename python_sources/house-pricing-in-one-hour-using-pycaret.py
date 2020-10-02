#!/usr/bin/env python
# coding: utf-8

# # <font color='green'> PyCaret - Machine Learning Made Easy</font>
# 
# ***3 things why I like PyCaret,***
# ## <font color='brown'> SuperEasy - SuperFast - SuperPowerful </font>
# 
# ### PyCaret is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within seconds in your choice of notebook environment.
# 
# 
# 
# PyCaret runs on Jupyter Notebook, Google Colab and Kaggle. So, it is platform friendly. 
# 
# Get PyCaret it from: https://pycaret.org/
# Learn PyCaret  from: https://pycaret.org/tutorial/
# 
# If you are a beginer, start here https://colab.research.google.com/drive/1GqQ3XAIzg4krBbnOpKyeRqT0qBQhdwYL
# 
# Or if you are good enough, go to https://github.com/pycaret/pycaret/blob/master/Tutorials/Binary%20Classification%20Tutorial%20Level%20Intermediate%20-%20CLF102.ipynb
# 
# **Great Learning !**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Install PyCaret Library

# In[ ]:


get_ipython().system('pip install pycaret')
from pycaret.regression import *


# # Get Data

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub   = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print('Training Data = ',train.shape)
print('Testing Data = ',test.shape)
train.head(3)


# ## Handle Missing Values in Training and Testing Dataset

# In[ ]:


def impute_missing_values(df):
    
    # Get Continuous and Categorical Features
    missing_cols = df.columns[df.isna().sum()>0]
    mean_cols = df[missing_cols].describe().columns
    mode_cols = list((set(missing_cols) - set(mean_cols)))

    # Impute Missing Values
    for col in mean_cols:
        df[col].fillna(df[col].mean(),axis=0,inplace=True)    

    for col in mode_cols:
        df[col].fillna('Unknown',axis=0,inplace=True)
        
    return df

train = impute_missing_values(train)
test  = impute_missing_values(test)

print('\nTraining Set after Imputation\nShape=',train.shape,'\nMissing Values=',train.isna().sum().sum())    
print('\nTesting Set after Imputation\nShape=',test.shape,'\nMissing Values=',test.isna().sum().sum())    


# ### Uncommon Columns
# - Target Variable

# In[ ]:


A = set(train.columns)
B = set(test.columns)

print('Uncommon Columns are ',A.union(B) - A.intersection(B))


# ## Data Setup

# In[ ]:


clf1 = setup(data       = train, 
             target     = 'SalePrice',             
             session_id = 123)


# ## Finding the Suitable Model 

# In[ ]:


compare_models(sort='MSE', fold=2)


# # CATBoost

# In[ ]:


cat = create_model('catboost',fold=10)
tune_cat = tune_model('catboost',fold=10,optimize='mse')


# In[ ]:


interpret_model(cat)


# In[ ]:


interpret_model(tune_cat)


# # XGBoost

# In[ ]:


xgb = create_model('xgboost',fold=10)


# In[ ]:


tune_xgb = tune_model('xgboost',fold=10,optimize='mse')


# In[ ]:


plot_model(xgb,plot='residuals')


# In[ ]:


plot_model(xgb,plot='error')


# In[ ]:


plot_model(xgb,plot='feature')


# In[ ]:


plot_model(tune_xgb,plot='residuals')


# In[ ]:


plot_model(tune_xgb, plot='error')


# In[ ]:


plot_model(tune_xgb,plot='feature')


# # Random Forest

# In[ ]:


rf = create_model('rf',fold=10)


# In[ ]:


tune_rf = tune_model('rf',fold=10,optimize='mse')


# In[ ]:


plot_model(rf, plot='residuals')


# In[ ]:


plot_model(tune_rf, plot='residuals')


# In[ ]:


plot_model(rf, plot='error')


# In[ ]:


plot_model(tune_rf, plot='error')


# In[ ]:


plot_model(rf, plot='feature')


# In[ ]:


plot_model(tune_rf, plot='feature')


# # Predict Test Set

# In[ ]:


xgb_pred = predict_model(xgb, data=test)
xgb_pred.head(3)


# In[ ]:


tune_cat_pred = predict_model(tune_cat, data=test)
tune_cat_pred.head(3)


# ## Save Output Result

# In[ ]:


sub1 = xgb_pred[['Id','Label']]
sub1.columns = ['Id','SalePrice']
sub1.to_csv('submission_xgb.csv',index=False)
sub1.head()


# In[ ]:


sub2 = tune_cat_pred[['Id','Label']]
sub2.columns = ['Id','SalePrice']
sub2.to_csv('submission_tune_cat.csv',index=False)
sub2.head()


# In[ ]:




